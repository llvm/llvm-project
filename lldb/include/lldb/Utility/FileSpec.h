//===-- FileSpec.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_FileSpec_h_
#define liblldb_FileSpec_h_

// C Includes
// C++ Includes
#include <functional>
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/Utility/ConstString.h"

#include "llvm/ADT/StringRef.h" // for StringRef
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t, uint64_t

namespace lldb_private {
class Stream;
}
namespace llvm {
class Triple;
}
namespace llvm {
class raw_ostream;
}
namespace llvm {
template <typename T> class SmallVectorImpl;
}

namespace lldb_private {

//----------------------------------------------------------------------
/// @class FileSpec FileSpec.h "lldb/Host/FileSpec.h"
/// @brief A file utility class.
///
/// A file specification class that divides paths up into a directory
/// and basename. These string values of the paths are put into uniqued
/// string pools for fast comparisons and efficient memory usage.
///
/// Another reason the paths are split into the directory and basename
/// is to allow efficient debugger searching. Often in a debugger the
/// user types in the basename of the file, for example setting a
/// breakpoint by file and line, or specifying a module (shared library)
/// to limit the scope in which to execute a command. The user rarely
/// types in a full path. When the paths are already split up, it makes
/// it easy for us to compare only the basenames of a lot of file
/// specifications without having to split up the file path each time
/// to get to the basename.
//----------------------------------------------------------------------
class FileSpec {
public:
  enum PathSyntax {
    ePathSyntaxPosix,
    ePathSyntaxWindows,
    ePathSyntaxHostNative
  };

  FileSpec();

  //------------------------------------------------------------------
  /// Constructor with path.
  ///
  /// Takes a path to a file which can be just a filename, or a full
  /// path. If \a path is not nullptr or empty, this function will call
  /// FileSpec::SetFile (const char *path, bool resolve).
  ///
  /// @param[in] path
  ///     The full or partial path to a file.
  ///
  /// @param[in] resolve_path
  ///     If \b true, then we resolve the path, removing stray ../.. and so
  ///     forth,
  ///     if \b false we trust the path is in canonical form already.
  ///
  /// @see FileSpec::SetFile (const char *path, bool resolve)
  //------------------------------------------------------------------
  explicit FileSpec(llvm::StringRef path, bool resolve_path,
                    PathSyntax syntax = ePathSyntaxHostNative);

  explicit FileSpec(llvm::StringRef path, bool resolve_path,
                    const llvm::Triple &Triple);

  //------------------------------------------------------------------
  /// Copy constructor
  ///
  /// Makes a copy of the uniqued directory and filename strings from
  /// \a rhs.
  ///
  /// @param[in] rhs
  ///     A const FileSpec object reference to copy.
  //------------------------------------------------------------------
  FileSpec(const FileSpec &rhs);

  //------------------------------------------------------------------
  /// Copy constructor
  ///
  /// Makes a copy of the uniqued directory and filename strings from
  /// \a rhs if it is not nullptr.
  ///
  /// @param[in] rhs
  ///     A const FileSpec object pointer to copy if non-nullptr.
  //------------------------------------------------------------------
  FileSpec(const FileSpec *rhs);

  //------------------------------------------------------------------
  /// Destructor.
  //------------------------------------------------------------------
  ~FileSpec();

  bool DirectoryEquals(const FileSpec &other) const;

  bool FileEquals(const FileSpec &other) const;

  //------------------------------------------------------------------
  /// Assignment operator.
  ///
  /// Makes a copy of the uniqued directory and filename strings from
  /// \a rhs.
  ///
  /// @param[in] rhs
  ///     A const FileSpec object reference to assign to this object.
  ///
  /// @return
  ///     A const reference to this object.
  //------------------------------------------------------------------
  const FileSpec &operator=(const FileSpec &rhs);

  //------------------------------------------------------------------
  /// Equal to operator
  ///
  /// Tests if this object is equal to \a rhs.
  ///
  /// @param[in] rhs
  ///     A const FileSpec object reference to compare this object
  ///     to.
  ///
  /// @return
  ///     \b true if this object is equal to \a rhs, \b false
  ///     otherwise.
  //------------------------------------------------------------------
  bool operator==(const FileSpec &rhs) const;

  //------------------------------------------------------------------
  /// Not equal to operator
  ///
  /// Tests if this object is not equal to \a rhs.
  ///
  /// @param[in] rhs
  ///     A const FileSpec object reference to compare this object
  ///     to.
  ///
  /// @return
  ///     \b true if this object is equal to \a rhs, \b false
  ///     otherwise.
  //------------------------------------------------------------------
  bool operator!=(const FileSpec &rhs) const;

  //------------------------------------------------------------------
  /// Less than to operator
  ///
  /// Tests if this object is less than \a rhs.
  ///
  /// @param[in] rhs
  ///     A const FileSpec object reference to compare this object
  ///     to.
  ///
  /// @return
  ///     \b true if this object is less than \a rhs, \b false
  ///     otherwise.
  //------------------------------------------------------------------
  bool operator<(const FileSpec &rhs) const;

  //------------------------------------------------------------------
  /// Convert to pointer operator.
  ///
  /// This allows code to check a FileSpec object to see if it
  /// contains anything valid using code such as:
  ///
  /// @code
  /// FileSpec file_spec(...);
  /// if (file_spec)
  /// { ...
  /// @endcode
  ///
  /// @return
  ///     A pointer to this object if either the directory or filename
  ///     is valid, nullptr otherwise.
  //------------------------------------------------------------------
  explicit operator bool() const;

  //------------------------------------------------------------------
  /// Logical NOT operator.
  ///
  /// This allows code to check a FileSpec object to see if it is
  /// invalid using code such as:
  ///
  /// @code
  /// FileSpec file_spec(...);
  /// if (!file_spec)
  /// { ...
  /// @endcode
  ///
  /// @return
  ///     Returns \b true if the object has an empty directory and
  ///     filename, \b false otherwise.
  //------------------------------------------------------------------
  bool operator!() const;

  //------------------------------------------------------------------
  /// Clears the object state.
  ///
  /// Clear this object by releasing both the directory and filename
  /// string values and reverting them to empty strings.
  //------------------------------------------------------------------
  void Clear();

  //------------------------------------------------------------------
  /// Compare two FileSpec objects.
  ///
  /// If \a full is true, then both the directory and the filename
  /// must match. If \a full is false, then the directory names for
  /// \a lhs and \a rhs are only compared if they are both not empty.
  /// This allows a FileSpec object to only contain a filename
  /// and it can match FileSpec objects that have matching
  /// filenames with different paths.
  ///
  /// @param[in] lhs
  ///     A const reference to the Left Hand Side object to compare.
  ///
  /// @param[in] rhs
  ///     A const reference to the Right Hand Side object to compare.
  ///
  /// @param[in] full
  ///     If true, then both the directory and filenames will have to
  ///     match for a compare to return zero (equal to). If false
  ///     and either directory from \a lhs or \a rhs is empty, then
  ///     only the filename will be compared, else a full comparison
  ///     is done.
  ///
  /// @return
  ///     @li -1 if \a lhs is less than \a rhs
  ///     @li 0 if \a lhs is equal to \a rhs
  ///     @li 1 if \a lhs is greater than \a rhs
  //------------------------------------------------------------------
  static int Compare(const FileSpec &lhs, const FileSpec &rhs, bool full);

  static bool Equal(const FileSpec &a, const FileSpec &b, bool full,
                    bool remove_backups = false);

  //------------------------------------------------------------------
  /// Case sensitivity of path.
  ///
  /// @return
  ///     \b true if the file path is case sensitive (POSIX), false
  ///		if case insensitive (Windows).
  //------------------------------------------------------------------
  bool IsCaseSensitive() const { return m_syntax != ePathSyntaxWindows; }

  //------------------------------------------------------------------
  /// Dump this object to a Stream.
  ///
  /// Dump the object to the supplied stream \a s. If the object
  /// contains a valid directory name, it will be displayed followed
  /// by a directory delimiter, and the filename.
  ///
  /// @param[in] s
  ///     The stream to which to dump the object description.
  //------------------------------------------------------------------
  void Dump(Stream *s) const;

  //------------------------------------------------------------------
  /// Existence test.
  ///
  /// @return
  ///     \b true if the file exists on disk, \b false otherwise.
  //------------------------------------------------------------------
  bool Exists() const;

  //------------------------------------------------------------------
  /// Check if a file is readable by the current user
  ///
  /// @return
  ///     \b true if the file exists on disk and is readable, \b false
  ///     otherwise.
  //------------------------------------------------------------------
  bool Readable() const;

  //------------------------------------------------------------------
  /// Expanded existence test.
  ///
  /// Call into the Host to see if it can help find the file (e.g. by
  /// searching paths set in the environment, etc.).
  ///
  /// If found, sets the value of m_directory to the directory where
  /// the file was found.
  ///
  /// @return
  ///     \b true if was able to find the file using expanded search
  ///     methods, \b false otherwise.
  //------------------------------------------------------------------
  bool ResolveExecutableLocation();

  //------------------------------------------------------------------
  /// Canonicalize this file path (basically running the static
  /// FileSpec::Resolve method on it). Useful if you asked us not to
  /// resolve the file path when you set the file.
  //------------------------------------------------------------------
  bool ResolvePath();

  uint64_t GetByteSize() const;

  PathSyntax GetPathSyntax() const;

  //------------------------------------------------------------------
  /// Directory string get accessor.
  ///
  /// @return
  ///     A reference to the directory string object.
  //------------------------------------------------------------------
  ConstString &GetDirectory();

  //------------------------------------------------------------------
  /// Directory string const get accessor.
  ///
  /// @return
  ///     A const reference to the directory string object.
  //------------------------------------------------------------------
  const ConstString &GetDirectory() const;

  //------------------------------------------------------------------
  /// Filename string get accessor.
  ///
  /// @return
  ///     A reference to the filename string object.
  //------------------------------------------------------------------
  ConstString &GetFilename();

  //------------------------------------------------------------------
  /// Filename string const get accessor.
  ///
  /// @return
  ///     A const reference to the filename string object.
  //------------------------------------------------------------------
  const ConstString &GetFilename() const;

  //------------------------------------------------------------------
  /// Returns true if the filespec represents an implementation source
  /// file (files with a ".c", ".cpp", ".m", ".mm" (many more)
  /// extension).
  ///
  /// @return
  ///     \b true if the filespec represents an implementation source
  ///     file, \b false otherwise.
  //------------------------------------------------------------------
  bool IsSourceImplementationFile() const;

  //------------------------------------------------------------------
  /// Returns true if the filespec represents a relative path.
  ///
  /// @return
  ///     \b true if the filespec represents a relative path,
  ///     \b false otherwise.
  //------------------------------------------------------------------
  bool IsRelative() const;

  //------------------------------------------------------------------
  /// Returns true if the filespec represents an absolute path.
  ///
  /// @return
  ///     \b true if the filespec represents an absolute path,
  ///     \b false otherwise.
  //------------------------------------------------------------------
  bool IsAbsolute() const;

  //------------------------------------------------------------------
  /// Extract the full path to the file.
  ///
  /// Extract the directory and path into a fixed buffer. This is
  /// needed as the directory and path are stored in separate string
  /// values.
  ///
  /// @param[out] path
  ///     The buffer in which to place the extracted full path.
  ///
  /// @param[in] max_path_length
  ///     The maximum length of \a path.
  ///
  /// @return
  ///     Returns the number of characters that would be needed to
  ///     properly copy the full path into \a path. If the returned
  ///     number is less than \a max_path_length, then the path is
  ///     properly copied and terminated. If the return value is
  ///     >= \a max_path_length, then the path was truncated (but is
  ///     still NULL terminated).
  //------------------------------------------------------------------
  size_t GetPath(char *path, size_t max_path_length,
                 bool denormalize = true) const;

  //------------------------------------------------------------------
  /// Extract the full path to the file.
  ///
  /// Extract the directory and path into a std::string, which is returned.
  ///
  /// @return
  ///     Returns a std::string with the directory and filename
  ///     concatenated.
  //------------------------------------------------------------------
  std::string GetPath(bool denormalize = true) const;

  const char *GetCString(bool denormalize = true) const;

  //------------------------------------------------------------------
  /// Extract the full path to the file.
  ///
  /// Extract the directory and path into an llvm::SmallVectorImpl<>
  ///
  /// @return
  ///     Returns a std::string with the directory and filename
  ///     concatenated.
  //------------------------------------------------------------------
  void GetPath(llvm::SmallVectorImpl<char> &path,
               bool denormalize = true) const;

  //------------------------------------------------------------------
  /// Extract the extension of the file.
  ///
  /// Returns a ConstString that represents the extension of the filename
  /// for this FileSpec object. If this object does not represent a file,
  /// or the filename has no extension, ConstString(nullptr) is returned.
  /// The dot ('.') character is not returned as part of the extension
  ///
  /// @return
  ///     Returns the extension of the file as a ConstString object.
  //------------------------------------------------------------------
  ConstString GetFileNameExtension() const;

  //------------------------------------------------------------------
  /// Return the filename without the extension part
  ///
  /// Returns a ConstString that represents the filename of this object
  /// without the extension part (e.g. for a file named "foo.bar", "foo"
  /// is returned)
  ///
  /// @return
  ///     Returns the filename without extension
  ///     as a ConstString object.
  //------------------------------------------------------------------
  ConstString GetFileNameStrippingExtension() const;

  //------------------------------------------------------------------
  /// Return the current permissions of the path.
  ///
  /// Returns a bitmask for the current permissions of the file
  /// ( zero or more of the permission bits defined in
  /// File::Permissions).
  ///
  /// @return
  ///     Zero if the file doesn't exist or we are unable to get
  ///     information for the file, otherwise one or more permission
  ///     bits from the File::Permissions enumeration.
  //------------------------------------------------------------------
  uint32_t GetPermissions() const;

  //------------------------------------------------------------------
  /// Get the memory cost of this object.
  ///
  /// Return the size in bytes that this object takes in memory. This
  /// returns the size in bytes of this object, not any shared string
  /// values it may refer to.
  ///
  /// @return
  ///     The number of bytes that this object occupies in memory.
  ///
  /// @see ConstString::StaticMemorySize ()
  //------------------------------------------------------------------
  size_t MemorySize() const;

  //------------------------------------------------------------------
  /// Normalize a pathname by collapsing redundant separators and
  /// up-level references.
  //------------------------------------------------------------------
  FileSpec GetNormalizedPath() const;

  //------------------------------------------------------------------
  /// Change the file specified with a new path.
  ///
  /// Update the contents of this object with a new path. The path will
  /// be split up into a directory and filename and stored as uniqued
  /// string values for quick comparison and efficient memory usage.
  ///
  /// @param[in] path
  ///     A full, partial, or relative path to a file.
  ///
  /// @param[in] resolve_path
  ///     If \b true, then we will try to resolve links the path using
  ///     the static FileSpec::Resolve.
  //------------------------------------------------------------------
  void SetFile(llvm::StringRef path, bool resolve_path,
               PathSyntax syntax = ePathSyntaxHostNative);

  void SetFile(llvm::StringRef path, bool resolve_path,
               const llvm::Triple &Triple);

  bool IsResolved() const { return m_is_resolved; }

  //------------------------------------------------------------------
  /// Set if the file path has been resolved or not.
  ///
  /// If you know a file path is already resolved and avoided passing
  /// a \b true parameter for any functions that take a "bool
  /// resolve_path" parameter, you can set the value manually using
  /// this call to make sure we don't try and resolve it later, or try
  /// and resolve a path that has already been resolved.
  ///
  /// @param[in] is_resolved
  ///     A boolean value that will replace the current value that
  ///     indicates if the paths in this object have been resolved.
  //------------------------------------------------------------------
  void SetIsResolved(bool is_resolved) { m_is_resolved = is_resolved; }

  //------------------------------------------------------------------
  /// Resolves user name and links in \a path, and overwrites the input
  /// argument with the resolved path.
  ///
  /// @param[in] path
  ///     Input path to be resolved, in the form of a llvm::SmallString or
  ///     similar.
  //------------------------------------------------------------------
  static void Resolve(llvm::SmallVectorImpl<char> &path);

  FileSpec CopyByAppendingPathComponent(llvm::StringRef component) const;
  FileSpec CopyByRemovingLastPathComponent() const;

  void PrependPathComponent(llvm::StringRef component);
  void PrependPathComponent(const FileSpec &new_path);

  void AppendPathComponent(llvm::StringRef component);
  void AppendPathComponent(const FileSpec &new_path);

  void RemoveLastPathComponent();

  ConstString GetLastPathComponent() const;

  enum EnumerateDirectoryResult {
    eEnumerateDirectoryResultNext,  // Enumerate next entry in the current
                                    // directory
    eEnumerateDirectoryResultEnter, // Recurse into the current entry if it is a
                                    // directory or symlink, or next if not
    eEnumerateDirectoryResultQuit   // Stop directory enumerations at any level
  };

  typedef EnumerateDirectoryResult (*EnumerateDirectoryCallbackType)(
      void *baton, llvm::sys::fs::file_type file_type, const FileSpec &spec);

  static void EnumerateDirectory(llvm::StringRef dir_path,
                                 bool find_directories, bool find_files,
                                 bool find_other,
                                 EnumerateDirectoryCallbackType callback,
                                 void *callback_baton);

  typedef std::function<EnumerateDirectoryResult(
      llvm::sys::fs::file_type file_type, const FileSpec &spec)>
      DirectoryCallback;

protected:
  //------------------------------------------------------------------
  // Member variables
  //------------------------------------------------------------------
  ConstString m_directory;            ///< The uniqued directory path
  ConstString m_filename;             ///< The uniqued filename path
  mutable bool m_is_resolved = false; ///< True if this path has been resolved.
  PathSyntax
      m_syntax; ///< The syntax that this path uses (e.g. Windows / Posix)
};

//----------------------------------------------------------------------
/// Dump a FileSpec object to a stream
//----------------------------------------------------------------------
Stream &operator<<(Stream &s, const FileSpec &f);

} // namespace lldb_private

namespace llvm {

/// Implementation of format_provider<T> for FileSpec.
///
/// The options string of a FileSpec has the grammar:
///
///   file_spec_options   :: (empty) | F | D
///
///   =======================================================
///   |  style  |     Meaning          |      Example       |
///   -------------------------------------------------------
///   |         |                      |  Input   |  Output |
///   =======================================================
///   |    F    | Only print filename  | /foo/bar |   bar   |
///   |    D    | Only print directory | /foo/bar |  /foo/  |
///   | (empty) | Print file and dir   |          |         |
///   =======================================================
///
/// Any other value is considered an invalid format string.
///
template <> struct format_provider<lldb_private::FileSpec> {
  static void format(const lldb_private::FileSpec &F, llvm::raw_ostream &Stream,
                     StringRef Style);
};
}

#endif // liblldb_FileSpec_h_
