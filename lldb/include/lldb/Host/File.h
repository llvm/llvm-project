//===-- File.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_File_h_
#define liblldb_File_h_
#if defined(__cplusplus)

#include <stdio.h>

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class File File.h "lldb/Host/File.h"
/// @brief A file class.
///
/// A file class that divides abstracts the LLDB core from host file
/// functionality.
//----------------------------------------------------------------------
class File
{
public:
    static int kInvalidDescriptor;
    static FILE * kInvalidStream;

    enum OpenOptions
    {
        eOpenOptionRead                 = (1u << 0),    // Open file for reading
        eOpenOptionWrite                = (1u << 1),    // Open file for writing
        eOpenOptionAppend               = (1u << 2),    // Don't truncate file when opening, append to end of file
        eOpenOptionTruncate             = (1u << 3),    // Truncate file when opening
        eOpenOptionNonBlocking          = (1u << 4),    // File reads
        eOpenOptionCanCreate            = (1u << 5),    // Create file if doesn't already exist
        eOpenOptionCanCreateNewOnly     = (1u << 6)     // Can create file only if it doesn't already exist
    };
    
    enum Permissions
    {
        ePermissionsUserRead        = (1u << 0),
        ePermissionsUserWrite       = (1u << 1),
        ePermissionsUserExecute     = (1u << 2),
        ePermissionsGroupRead       = (1u << 3),
        ePermissionsGroupWrite      = (1u << 4),
        ePermissionsGroupExecute    = (1u << 5),
        ePermissionsWorldRead       = (1u << 6),
        ePermissionsWorldWrite      = (1u << 7),
        ePermissionsWorldExecute    = (1u << 8),

        ePermissionsUserRW      = (ePermissionsUserRead    | ePermissionsUserWrite    | 0                        ),
        ePermissionsUserRX      = (ePermissionsUserRead    | 0                        | ePermissionsUserExecute  ),
        ePermissionsUserRWX     = (ePermissionsUserRead    | ePermissionsUserWrite    | ePermissionsUserExecute  ),

        ePermissionsGroupRW     = (ePermissionsGroupRead   | ePermissionsGroupWrite   | 0                        ),
        ePermissionsGroupRX     = (ePermissionsGroupRead   | 0                        | ePermissionsGroupExecute ),
        ePermissionsGroupRWX    = (ePermissionsGroupRead   | ePermissionsGroupWrite   | ePermissionsGroupExecute ),

        ePermissionsWorldRW     = (ePermissionsWorldRead   | ePermissionsWorldWrite   | 0                        ),
        ePermissionsWorldRX     = (ePermissionsWorldRead   | 0                        | ePermissionsWorldExecute ),
        ePermissionsWorldRWX    = (ePermissionsWorldRead   | ePermissionsWorldWrite   | ePermissionsWorldExecute ),

        ePermissionsEveryoneR   = (ePermissionsUserRead    | ePermissionsGroupRead    | ePermissionsWorldRead    ),
        ePermissionsEveryoneW   = (ePermissionsUserWrite   | ePermissionsGroupWrite   | ePermissionsWorldWrite   ),
        ePermissionsEveryoneX   = (ePermissionsUserExecute | ePermissionsGroupExecute | ePermissionsWorldExecute ),

        ePermissionsEveryoneRW  = (ePermissionsEveryoneR   | ePermissionsEveryoneW    | 0                        ),
        ePermissionsEveryoneRX  = (ePermissionsEveryoneR   | 0                        | ePermissionsEveryoneX    ),
        ePermissionsEveryoneRWX = (ePermissionsEveryoneR   | ePermissionsEveryoneW    | ePermissionsEveryoneX    ),
        ePermissionsDefault     = (ePermissionsUserRW      | ePermissionsGroupRead)
    };

    File() : 
        m_descriptor (kInvalidDescriptor),
        m_stream (kInvalidStream),
        m_options (0),
        m_owned (false)
    {
    }
    
    File (FILE *fh, bool transfer_ownership) :
        m_descriptor (kInvalidDescriptor),
        m_stream (fh),
        m_options (0),
        m_owned (transfer_ownership)
    {
    }

    File (const File &rhs);
    
    File &
    operator= (const File &rhs);
    //------------------------------------------------------------------
    /// Constructor with path.
    ///
    /// Takes a path to a file which can be just a filename, or a full
    /// path. If \a path is not NULL or empty, this function will call
    /// File::Open (const char *path, uint32_t options, uint32_t permissions).
    ///
    /// @param[in] path
    ///     The full or partial path to a file.
    ///
    /// @param[in] options
    ///     Options to use when opening (see File::OpenOptions)
    ///
    /// @param[in] permissions
    ///     Options to use when opening (see File::Permissions)
    ///
    /// @see File::Open (const char *path, uint32_t options, uint32_t permissions)
    //------------------------------------------------------------------
    File (const char *path,
          uint32_t options,
          uint32_t permissions = ePermissionsDefault);

    
    File (int fd, bool tranfer_ownership) : 
        m_descriptor (fd),
        m_stream (kInvalidStream),
        m_options (0),
        m_owned (tranfer_ownership)
    {
    }
    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~File ();

    bool
    IsValid () const
    {
        return DescriptorIsValid() || StreamIsValid();
    }

    //------------------------------------------------------------------
    /// Convert to pointer operator.
    ///
    /// This allows code to check a File object to see if it
    /// contains anything valid using code such as:
    ///
    /// @code
    /// File file(...);
    /// if (file)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     A pointer to this object if either the directory or filename
    ///     is valid, NULL otherwise.
    //------------------------------------------------------------------
    operator
    bool () const
    {
        return DescriptorIsValid() || StreamIsValid();
    }

    //------------------------------------------------------------------
    /// Logical NOT operator.
    ///
    /// This allows code to check a File object to see if it is
    /// invalid using code such as:
    ///
    /// @code
    /// File file(...);
    /// if (!file)
    /// { ...
    /// @endcode
    ///
    /// @return
    ///     Returns \b true if the object has an empty directory and
    ///     filename, \b false otherwise.
    //------------------------------------------------------------------
    bool
    operator! () const
    {
        return !DescriptorIsValid() && !StreamIsValid();
    }

    //------------------------------------------------------------------
    /// Get the file spec for this file.
    ///
    /// @return
    ///     A reference to the file specification object.
    //------------------------------------------------------------------
    Error
    GetFileSpec (FileSpec &file_spec) const;
    
    //------------------------------------------------------------------
    /// Open a file for read/writing with the specified options.
    ///
    /// Takes a path to a file which can be just a filename, or a full
    /// path.
    ///
    /// @param[in] path
    ///     The full or partial path to a file.
    ///
    /// @param[in] options
    ///     Options to use when opening (see File::OpenOptions)
    ///
    /// @param[in] permissions
    ///     Options to use when opening (see File::Permissions)
    //------------------------------------------------------------------
    Error
    Open (const char *path,
          uint32_t options,
          uint32_t permissions = ePermissionsDefault);

    Error
    Close ();
    
    Error
    Duplicate (const File &rhs);

    int
    GetDescriptor() const;

    void
    SetDescriptor(int fd, bool transfer_ownership);

    FILE *
    GetStream ();

    void
    SetStream (FILE *fh, bool transfer_ownership);

    //------------------------------------------------------------------
    /// Read bytes from a file from the current file position.
    ///
    /// NOTE: This function is NOT thread safe. Use the read function
    /// that takes an "off_t &offset" to ensure correct operation in
    /// multi-threaded environments.
    ///
    /// @param[in] buf
    ///     A buffer where to put the bytes that are read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to read form the current file position
    ///     which gets modified with the number of bytes that were read.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Read (void *buf, size_t &num_bytes);
          
    //------------------------------------------------------------------
    /// Write bytes to a file at the current file position.
    ///
    /// NOTE: This function is NOT thread safe. Use the write function
    /// that takes an "off_t &offset" to ensure correct operation in
    /// multi-threaded environments.
    ///
    /// @param[in] buf
    ///     A buffer where to put the bytes that are read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to write to the current file position
    ///     which gets modified with the number of bytes that were 
    ///     written.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Write (const void *buf, size_t &num_bytes);

    //------------------------------------------------------------------
    /// Seek to an offset relative to the beginning of the file.
    ///
    /// NOTE: This function is NOT thread safe, other threads that 
    /// access this object might also change the current file position.
    /// For thread safe reads and writes see the following functions:
    /// @see File::Read (void *, size_t, off_t &)
    /// @see File::Write (const void *, size_t, off_t &)
    ///
    /// @param[in] offset
    ///     The offset to seek to within the file relative to the 
    ///     beginning of the file.
    ///
    /// @param[in] error_ptr
    ///     A pointer to a lldb_private::Error object that will be
    ///     filled in if non-NULL.
    ///
    /// @return
    ///     The resulting seek offset, or -1 on error.
    //------------------------------------------------------------------
    off_t
    SeekFromStart (off_t offset, Error *error_ptr = NULL);
    
    //------------------------------------------------------------------
    /// Seek to an offset relative to the current file position.
    ///
    /// NOTE: This function is NOT thread safe, other threads that 
    /// access this object might also change the current file position.
    /// For thread safe reads and writes see the following functions:
    /// @see File::Read (void *, size_t, off_t &)
    /// @see File::Write (const void *, size_t, off_t &)
    ///
    /// @param[in] offset
    ///     The offset to seek to within the file relative to the 
    ///     current file position.
    ///
    /// @param[in] error_ptr
    ///     A pointer to a lldb_private::Error object that will be
    ///     filled in if non-NULL.
    ///
    /// @return
    ///     The resulting seek offset, or -1 on error.
    //------------------------------------------------------------------
    off_t
    SeekFromCurrent (off_t offset, Error *error_ptr = NULL);
    
    //------------------------------------------------------------------
    /// Seek to an offset relative to the end of the file.
    ///
    /// NOTE: This function is NOT thread safe, other threads that 
    /// access this object might also change the current file position.
    /// For thread safe reads and writes see the following functions:
    /// @see File::Read (void *, size_t, off_t &)
    /// @see File::Write (const void *, size_t, off_t &)
    ///
    /// @param[in/out] offset
    ///     The offset to seek to within the file relative to the 
    ///     end of the file which gets filled in the the resulting
    ///     absolute file offset.
    ///
    /// @param[in] error_ptr
    ///     A pointer to a lldb_private::Error object that will be
    ///     filled in if non-NULL.
    ///
    /// @return
    ///     The resulting seek offset, or -1 on error.
    //------------------------------------------------------------------
    off_t
    SeekFromEnd (off_t offset, Error *error_ptr = NULL);

    //------------------------------------------------------------------
    /// Read bytes from a file from the specified file offset.
    ///
    /// NOTE: This function is thread safe in that clients manager their
    /// own file position markers and reads on other threads won't mess
    /// up the current read.
    ///
    /// @param[in] buf
    ///     A buffer where to put the bytes that are read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to read form the current file position
    ///     which gets modified with the number of bytes that were read.
    ///
    /// @param[in/out] offset
    ///     The offset within the file from which to read \a num_bytes
    ///     bytes. This offset gets incremented by the number of bytes
    ///     that were read.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Read (void *dst, size_t &num_bytes, off_t &offset);
    
    //------------------------------------------------------------------
    /// Read bytes from a file from the specified file offset.
    ///
    /// NOTE: This function is thread safe in that clients manager their
    /// own file position markers and reads on other threads won't mess
    /// up the current read.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to read form the current file position
    ///     which gets modified with the number of bytes that were read.
    ///
    /// @param[in/out] offset
    ///     The offset within the file from which to read \a num_bytes
    ///     bytes. This offset gets incremented by the number of bytes
    ///     that were read.
    ///
    /// @param[in] null_terminate
    ///     Ensure that the data that is read is terminated with a NULL
    ///     character so that the data can be used as a C string.
    ///
    /// @param[out] data_buffer_sp
    ///     A data buffer to create and fill in that will contain any
    ///     data that is read from the file. This buffer will be reset
    ///     if an error occurs.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Read (size_t &num_bytes,
          off_t &offset,
          bool null_terminate,
          lldb::DataBufferSP &data_buffer_sp);

    //------------------------------------------------------------------
    /// Write bytes to a file at the specified file offset.
    ///
    /// NOTE: This function is thread safe in that clients manager their
    /// own file position markers, though clients will need to implement
    /// their own locking externally to avoid multiple people writing
    /// to the file at the same time.
    ///
    /// @param[in] buf
    ///     A buffer containing the bytes to write.
    ///
    /// @param[in/out] num_bytes
    ///     The number of bytes to write to the file at offset \a offset.
    ///     \a num_bytes gets modified with the number of bytes that 
    ///     were read.
    ///
    /// @param[in/out] offset
    ///     The offset within the file at which to write \a num_bytes
    ///     bytes. This offset gets incremented by the number of bytes
    ///     that were written.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Write (const void *src, size_t &num_bytes, off_t &offset);

    //------------------------------------------------------------------
    /// Flush the current stream
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Flush ();
    
    //------------------------------------------------------------------
    /// Sync to disk.
    ///
    /// @return
    ///     An error object that indicates success or the reason for 
    ///     failure.
    //------------------------------------------------------------------
    Error
    Sync ();

    //------------------------------------------------------------------
    /// Output printf formatted output to the stream.
    ///
    /// Print some formatted output to the stream.
    ///
    /// @param[in] format
    ///     A printf style format string.
    ///
    /// @param[in] ...
    ///     Variable arguments that are needed for the printf style
    ///     format string \a format.
    //------------------------------------------------------------------
    size_t
    Printf (const char *format, ...)  __attribute__ ((format (printf, 2, 3)));
    
    size_t
    PrintfVarArg(const char *format, va_list args);

protected:
    
    
    bool
    DescriptorIsValid () const
    {
        return m_descriptor >= 0;
    }

    bool
    StreamIsValid () const
    {
        return m_stream != kInvalidStream;
    }
    
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    int m_descriptor;
    FILE *m_stream;
    uint32_t m_options;
    bool m_owned;
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_File_h_
