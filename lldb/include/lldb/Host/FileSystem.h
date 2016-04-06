//===-- FileSystem.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_FileSystem_h
#define liblldb_Host_FileSystem_h

#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

#include "lldb/lldb-types.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/FileSpec.h"

namespace lldb_private
{
class FileSystem
{
  public:
    static const char *DEV_NULL;
    static const char *PATH_CONVERSION_ERROR;

    static FileSpec::PathSyntax GetNativePathSyntax();

    static Error MakeDirectory(const FileSpec &file_spec, uint32_t mode);
    static Error DeleteDirectory(const FileSpec &file_spec, bool recurse);

    static Error GetFilePermissions(const FileSpec &file_spec,
                                    uint32_t &file_permissions);
    static Error SetFilePermissions(const FileSpec &file_spec,
                                    uint32_t file_permissions);
    static lldb::user_id_t GetFileSize(const FileSpec &file_spec);
    static bool GetFileExists(const FileSpec &file_spec);

    static Error Hardlink(const FileSpec &src, const FileSpec &dst);
    static int GetHardlinkCount(const FileSpec &file_spec);
    static Error Symlink(const FileSpec &src, const FileSpec &dst);
    static Error Readlink(const FileSpec &src, FileSpec &dst);
    static Error Unlink(const FileSpec &file_spec);
    
    static Error ResolveSymbolicLink(const FileSpec &src, FileSpec &dst);

    static bool CalculateMD5(const FileSpec &file_spec, uint64_t &low, uint64_t &high);
    static bool CalculateMD5(const FileSpec &file_spec,
                             uint64_t offset,
                             uint64_t length,
                             uint64_t &low,
                             uint64_t &high);

    static bool CalculateMD5AsString(const FileSpec &file_spec, std::string& digest_str);
    static bool CalculateMD5AsString(const FileSpec &file_spec,
                                     uint64_t offset,
                                     uint64_t length,
                                     std::string& digest_str);

    /// Return \b true if \a spec is on a locally mounted file system, \b false otherwise.
    static bool IsLocal(const FileSpec &spec);

    /// Wraps ::fopen in a platform-independent way. Once opened, FILEs can be
    /// manipulated and closed with the normal ::fread, ::fclose, etc. functions.
    static FILE *
    Fopen(const char *path, const char *mode);

    /// Wraps ::stat in a platform-independent way.
    static int
    Stat(const char *path, struct stat *stats);
};
}

#endif
