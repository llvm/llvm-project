//===--- HeaderMap.cpp - A file that acts like dir of symlinks ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the HeaderMap interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderMap.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Data Structures and Manifest Constants
//===----------------------------------------------------------------------===//

enum {
  HMAP_HeaderMagicNumber = ('h' << 24) | ('m' << 16) | ('a' << 8) | 'p',
  HMAP_HeaderVersion = 1,

  HMAP_EmptyBucketKey = 0
};

namespace clang {
struct HMapBucket {
  uint32_t Key;          // Offset (into strings) of key.

  uint32_t Prefix;     // Offset (into strings) of value prefix.
  uint32_t Suffix;     // Offset (into strings) of value suffix.
};

struct HMapHeader {
  uint32_t Magic;           // Magic word, also indicates byte order.
  uint16_t Version;         // Version number -- currently 1.
  uint16_t Reserved;        // Reserved for future use - zero for now.
  uint32_t StringsOffset;   // Offset to start of string pool.
  uint32_t NumEntries;      // Number of entries in the string table.
  uint32_t NumBuckets;      // Number of buckets (always a power of 2).
  uint32_t MaxValueLength;  // Length of longest result path (excluding nul).
  // An array of 'NumBuckets' HMapBucket objects follows this header.
  // Strings follow the buckets, at StringsOffset.
};
} // end namespace clang.

/// HashHMapKey - This is the 'well known' hash function required by the file
/// format, used to look up keys in the hash table.  The hash table uses simple
/// linear probing based on this function.
static inline unsigned HashHMapKey(StringRef Str) {
  unsigned Result = 0;
  const char *S = Str.begin(), *End = Str.end();

  for (; S != End; S++)
    Result += tolower(*S) * 13;
  return Result;
}



//===----------------------------------------------------------------------===//
// Verification and Construction
//===----------------------------------------------------------------------===//

/// HeaderMap::Create - This attempts to load the specified file as a header
/// map.  If it doesn't look like a HeaderMap, it gives up and returns null.
/// If it looks like a HeaderMap but is obviously corrupted, it puts a reason
/// into the string error argument and returns null.
const HeaderMap *HeaderMap::Create(const FileEntry *FE, FileManager &FM) {
  // If the file is too small to be a header map, ignore it.
  unsigned FileSize = FE->getSize();
  if (FileSize <= sizeof(HMapHeader)) return 0;

  OwningPtr<const llvm::MemoryBuffer> FileBuffer(FM.getBufferForFile(FE));
  if (FileBuffer == 0) return 0;  // Unreadable file?
  const char *FileStart = FileBuffer->getBufferStart();

  // We know the file is at least as big as the header, check it now.
  const HMapHeader *Header = reinterpret_cast<const HMapHeader*>(FileStart);

  // Sniff it to see if it's a headermap by checking the magic number and
  // version.
  bool NeedsByteSwap;
  if (Header->Magic == HMAP_HeaderMagicNumber &&
      Header->Version == HMAP_HeaderVersion)
    NeedsByteSwap = false;
  else if (Header->Magic == llvm::ByteSwap_32(HMAP_HeaderMagicNumber) &&
           Header->Version == llvm::ByteSwap_16(HMAP_HeaderVersion))
    NeedsByteSwap = true;  // Mixed endianness headermap.
  else
    return 0;  // Not a header map.

  if (Header->Reserved != 0) return 0;

  // Okay, everything looks good, create the header map.
  return new HeaderMap(FileBuffer.take(), NeedsByteSwap);
}

HeaderMap::~HeaderMap() {
  delete FileBuffer;
}

//===----------------------------------------------------------------------===//
//  Utility Methods
//===----------------------------------------------------------------------===//


/// getFileName - Return the filename of the headermap.
const char *HeaderMap::getFileName() const {
  return FileBuffer->getBufferIdentifier();
}

unsigned HeaderMap::getEndianAdjustedWord(unsigned X) const {
  if (!NeedsBSwap) return X;
  return llvm::ByteSwap_32(X);
}

/// getHeader - Return a reference to the file header, in unbyte-swapped form.
/// This method cannot fail.
const HMapHeader &HeaderMap::getHeader() const {
  // We know the file is at least as big as the header.  Return it.
  return *reinterpret_cast<const HMapHeader*>(FileBuffer->getBufferStart());
}

/// getBucket - Return the specified hash table bucket from the header map,
/// bswap'ing its fields as appropriate.  If the bucket number is not valid,
/// this return a bucket with an empty key (0).
HMapBucket HeaderMap::getBucket(unsigned BucketNo) const {
  HMapBucket Result;
  Result.Key = HMAP_EmptyBucketKey;

  const HMapBucket *BucketArray =
    reinterpret_cast<const HMapBucket*>(FileBuffer->getBufferStart() +
                                        sizeof(HMapHeader));

  const HMapBucket *BucketPtr = BucketArray+BucketNo;
  if ((const char*)(BucketPtr+1) > FileBuffer->getBufferEnd()) {
    Result.Prefix = 0;
    Result.Suffix = 0;
    return Result;  // Invalid buffer, corrupt hmap.
  }

  // Otherwise, the bucket is valid.  Load the values, bswapping as needed.
  Result.Key    = getEndianAdjustedWord(BucketPtr->Key);
  Result.Prefix = getEndianAdjustedWord(BucketPtr->Prefix);
  Result.Suffix = getEndianAdjustedWord(BucketPtr->Suffix);
  return Result;
}

/// getString - Look up the specified string in the string table.  If the string
/// index is not valid, it returns an empty string.
const char *HeaderMap::getString(unsigned StrTabIdx) const {
  // Add the start of the string table to the idx.
  StrTabIdx += getEndianAdjustedWord(getHeader().StringsOffset);

  // Check for invalid index.
  if (StrTabIdx >= FileBuffer->getBufferSize())
    return 0;

  // Otherwise, we have a valid pointer into the file.  Just return it.  We know
  // that the "string" can not overrun the end of the file, because the buffer
  // is nul terminated by virtue of being a MemoryBuffer.
  return FileBuffer->getBufferStart()+StrTabIdx;
}

//===----------------------------------------------------------------------===//
// The Main Drivers
//===----------------------------------------------------------------------===//

/// dump - Print the contents of this headermap to stderr.
void HeaderMap::dump() const {
  const HMapHeader &Hdr = getHeader();
  unsigned NumBuckets = getEndianAdjustedWord(Hdr.NumBuckets);

  fprintf(stderr, "Header Map %s:\n  %d buckets, %d entries\n",
          getFileName(), NumBuckets,
          getEndianAdjustedWord(Hdr.NumEntries));

  for (unsigned i = 0; i != NumBuckets; ++i) {
    HMapBucket B = getBucket(i);
    if (B.Key == HMAP_EmptyBucketKey) continue;

    const char *Key    = getString(B.Key);
    const char *Prefix = getString(B.Prefix);
    const char *Suffix = getString(B.Suffix);
    fprintf(stderr, "  %d. %s -> '%s' '%s'\n", i, Key, Prefix, Suffix);
  }
}

/// LookupFile - Check to see if the specified relative filename is located in
/// this HeaderMap.  If so, open it and return its FileEntry.
const FileEntry *HeaderMap::LookupFile(
    StringRef Filename, FileManager &FM) const {
  const HMapHeader &Hdr = getHeader();
  unsigned NumBuckets = getEndianAdjustedWord(Hdr.NumBuckets);

  // If the number of buckets is not a power of two, the headermap is corrupt.
  // Don't probe infinitely.
  if (NumBuckets & (NumBuckets-1))
    return 0;

  // Linearly probe the hash table.
  for (unsigned Bucket = HashHMapKey(Filename);; ++Bucket) {
    HMapBucket B = getBucket(Bucket & (NumBuckets-1));
    if (B.Key == HMAP_EmptyBucketKey) return 0; // Hash miss.

    // See if the key matches.  If not, probe on.
    if (!Filename.equals_lower(getString(B.Key)))
      continue;

    // If so, we have a match in the hash table.  Construct the destination
    // path.
    SmallString<1024> DestPath;
    DestPath += getString(B.Prefix);
    DestPath += getString(B.Suffix);
    return FM.getFile(DestPath.str());
  }
}
