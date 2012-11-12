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
#if defined(__cplusplus)

#include "lldb/lldb-private.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/STLUtils.h"
#include "lldb/Host/TimeValue.h"

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
class FileSpec
{
public:
    typedef enum FileType
    {
        eFileTypeInvalid = -1,
        eFileTypeUnknown = 0,
        eFileTypeDirectory,
        eFileTypePipe,
        eFileTypeRegular,
        eFileTypeSocket,
        eFileTypeSymbolicLink,
        eFileTypeOther
    } FileType;

    FileSpec();

    //------------------------------------------------------------------
    /// Constructor with path.
    ///
    /// Takes a path to a file which can be just a filename, or a full
    /// path. If \a path is not NULL or empty, this function will call
    /// FileSpec::SetFile (const char *path, bool resolve).
    ///
    /// @param[in] path
    ///     The full or partial path to a file.
    ///
    /// @param[in] resolve_path
    ///     If \b true, then we resolve the path with realpath,
    ///     if \b false we trust the path is in canonical form already.
    ///
    /// @see FileSpec::SetFile (const char *path, bool resolve)
    //------------------------------------------------------------------
    explicit FileSpec (const char *path, bool resolve_path);

    //------------------------------------------------------------------
    /// Copy constructor
    ///
    /// Makes a copy of the uniqued directory and filename strings from
    /// \a rhs.
    ///
    /// @param[in] rhs
    ///     A const FileSpec object reference to copy.
    //------------------------------------------------------------------
    FileSpec (const FileSpec& rhs);

    //------------------------------------------------------------------
    /// Copy constructor
    ///
    /// Makes a copy of the uniqued directory and filename strings from
    /// \a rhs if it is not NULL.
    ///
    /// @param[in] rhs
    ///     A const FileSpec object pointer to copy if non-NULL.
    //------------------------------------------------------------------
    FileSpec (const FileSpec* rhs);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// The destructor is virtual in case this class is subclassed.
    //------------------------------------------------------------------
    virtual
    ~FileSpec ();

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
    const FileSpec&
    operator= (const FileSpec& rhs);

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
    bool
    operator== (const FileSpec& rhs) const;

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
    bool
    operator!= (const FileSpec& rhs) const;

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
    bool
    operator< (const FileSpec& rhs) const;

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
    ///     is valid, NULL otherwise.
    //------------------------------------------------------------------
    operator bool() const;

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
    bool
    operator! () const;

    //------------------------------------------------------------------
    /// Clears the object state.
    ///
    /// Clear this object by releasing both the directory and filename
    /// string values and reverting them to empty strings.
    //------------------------------------------------------------------
    void
    Clear ();

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
    static int
    Compare (const FileSpec& lhs, const FileSpec& rhs, bool full);

    static bool
    Equal (const FileSpec& a, const FileSpec& b, bool full);

    //------------------------------------------------------------------
    /// Dump this object to a Stream.
    ///
    /// Dump the object to the supplied stream \a s. If the object
    /// contains a valid directory name, it will be displayed followed
    /// by a directory delimiter, and the filename.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //------------------------------------------------------------------
    void
    Dump (Stream *s) const;

    //------------------------------------------------------------------
    /// Existence test.
    ///
    /// @return
    ///     \b true if the file exists on disk, \b false otherwise.
    //------------------------------------------------------------------
    bool
    Exists () const;

     
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
    bool
    ResolveExecutableLocation ();
    
    //------------------------------------------------------------------
    /// Canonicalize this file path (basically running the static 
    /// FileSpec::Resolve method on it). Useful if you asked us not to 
    /// resolve the file path when you set the file.
    //------------------------------------------------------------------
    bool
    ResolvePath ();

    uint64_t
    GetByteSize() const;

    //------------------------------------------------------------------
    /// Directory string get accessor.
    ///
    /// @return
    ///     A reference to the directory string object.
    //------------------------------------------------------------------
    ConstString &
    GetDirectory ();

    //------------------------------------------------------------------
    /// Directory string const get accessor.
    ///
    /// @return
    ///     A const reference to the directory string object.
    //------------------------------------------------------------------
    const ConstString &
    GetDirectory () const;

    //------------------------------------------------------------------
    /// Filename string get accessor.
    ///
    /// @return
    ///     A reference to the filename string object.
    //------------------------------------------------------------------
    ConstString &
    GetFilename ();

    //------------------------------------------------------------------
    /// Filename string const get accessor.
    ///
    /// @return
    ///     A const reference to the filename string object.
    //------------------------------------------------------------------
    const ConstString &
    GetFilename () const;

    //------------------------------------------------------------------
    /// Returns true if the filespec represents an implementation source
    /// file (files with a ".c", ".cpp", ".m", ".mm" (many more)
    /// extension).
    ///
    /// @return
    ///     \b true if the filespec represents an implementation source
    ///     file, \b false otherwise.
    //------------------------------------------------------------------
    bool
    IsSourceImplementationFile () const;

    //------------------------------------------------------------------
    /// Returns true if the filespec represents path that is relative
    /// path to the current working directory.
    ///
    /// @return
    ///     \b true if the filespec represents a current working
    ///     directory relative path, \b false otherwise.
    //------------------------------------------------------------------
    bool
    IsRelativeToCurrentWorkingDirectory () const;
    
    TimeValue
    GetModificationTime () const;

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
    size_t
    GetPath (char *path, size_t max_path_length) const;

    //------------------------------------------------------------------
    /// Extract the extension of the file.
    ///
    /// Returns a ConstString that represents the extension of the filename
    /// for this FileSpec object. If this object does not represent a file,
    /// or the filename has no extension, ConstString(NULL) is returned.
    /// The dot ('.') character is not returned as part of the extension
    ///
    /// @return
    ///     Returns the extension of the file as a ConstString object.
    //------------------------------------------------------------------
    ConstString
    GetFileNameExtension () const;

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
    ConstString
    GetFileNameStrippingExtension () const;
    
    FileType
    GetFileType () const;

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
    size_t
    MemorySize () const;

    //------------------------------------------------------------------
    /// Memory map part of, or the entire contents of, a file.
    ///
    /// Returns a shared pointer to a data buffer that contains all or
    /// part of the contents of a file. The data is memory mapped and
    /// will lazily page in data from the file as memory is accessed.
    /// The data that is mappped will start \a offset bytes into the
    /// file, and \a length bytes will be mapped. If \a length is
    /// greater than the number of bytes available in the file starting
    /// at \a offset, the number of bytes will be appropriately
    /// truncated. The final number of bytes that get mapped can be
    /// verified using the DataBuffer::GetByteSize() function on the return
    /// shared data pointer object contents.
    ///
    /// @param[in] offset
    ///     The offset in bytes from the beginning of the file where
    ///     memory mapping should begin.
    ///
    /// @param[in] length
    ///     The size in bytes that should be mapped starting \a offset
    ///     bytes into the file. If \a length is \c SIZE_MAX, map
    ///     as many bytes as possible.
    ///
    /// @return
    ///     A shared pointer to the memeory mapped data. This shared
    ///     pointer can contain a NULL DataBuffer pointer, so the contained
    ///     pointer must be checked prior to using it.
    //------------------------------------------------------------------
    lldb::DataBufferSP
    MemoryMapFileContents (off_t offset = 0, size_t length = SIZE_MAX) const;

    //------------------------------------------------------------------
    /// Read part of, or the entire contents of, a file into a heap based data buffer.
    ///
    /// Returns a shared pointer to a data buffer that contains all or
    /// part of the contents of a file. The data copies into a heap based
    /// buffer that lives in the DataBuffer shared pointer object returned.
    /// The data that is cached will start \a offset bytes into the
    /// file, and \a length bytes will be mapped. If \a length is
    /// greater than the number of bytes available in the file starting
    /// at \a offset, the number of bytes will be appropriately
    /// truncated. The final number of bytes that get mapped can be
    /// verified using the DataBuffer::GetByteSize() function.
    ///
    /// @param[in] offset
    ///     The offset in bytes from the beginning of the file where
    ///     memory mapping should begin.
    ///
    /// @param[in] length
    ///     The size in bytes that should be mapped starting \a offset
    ///     bytes into the file. If \a length is \c SIZE_MAX, map
    ///     as many bytes as possible.
    ///
    /// @return
    ///     A shared pointer to the memeory mapped data. This shared
    ///     pointer can contain a NULL DataBuffer pointer, so the contained
    ///     pointer must be checked prior to using it.
    //------------------------------------------------------------------
    lldb::DataBufferSP
    ReadFileContents (off_t offset = 0, size_t length = SIZE_MAX, Error *error_ptr = NULL) const;

    size_t
    ReadFileContents (off_t file_offset, void *dst, size_t dst_len, Error *error_ptr) const;

    
    //------------------------------------------------------------------
    /// Read the entire contents of a file as data that can be used
    /// as a C string.
    ///
    /// Read the entire contents of a file and ensure that the data
    /// is NULL terminated so it can be used as a C string.
    ///
    /// @return
    ///     A shared pointer to the data. This shared pointer can
    ///     contain a NULL DataBuffer pointer, so the contained pointer
    ///     must be checked prior to using it.
    //------------------------------------------------------------------
    lldb::DataBufferSP
    ReadFileContentsAsCString(Error *error_ptr = NULL);
    //------------------------------------------------------------------
    /// Change the file specificed with a new path.
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
    void
    SetFile (const char *path, bool resolve_path);

    bool
    IsResolved () const
    {
        return m_is_resolved;
    }

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
    void
    SetIsResolved (bool is_resolved)
    {
        m_is_resolved = is_resolved;
    }
    //------------------------------------------------------------------
    /// Read the file into an array of strings, one per line.
    ///
    /// Opens and reads the file in this object into an array of strings,
    /// one string per line of the file. Returns a boolean indicating
    /// success or failure.
    ///
    /// @param[out] lines
    ///     The string array into which to read the file.
    ///
    /// @result
    ///     Returns the number of lines that were read from the file.
    //------------------------------------------------------------------
    size_t
    ReadFileLines (STLStringArray &lines);

    //------------------------------------------------------------------
    /// Resolves user name and links in \a src_path, and writes the output
    /// to \a dst_path.  Note if the path pointed to by \a src_path does not
    /// exist, the contents of \a src_path will be copied to \a dst_path 
    /// unchanged.
    ///
    /// @param[in] src_path
    ///     Input path to be resolved.
    ///
    /// @param[in] dst_path
    ///     Buffer to store the resolved path.
    ///
    /// @param[in] dst_len 
    ///     Size of the buffer pointed to by dst_path.
    ///
    /// @result 
    ///     The number of characters required to write the resolved path.  If the
    ///     resolved path doesn't fit in dst_len, dst_len-1 characters will
    ///     be written to \a dst_path, but the actual required length will still be returned.
    //------------------------------------------------------------------
    static size_t
    Resolve (const char *src_path, char *dst_path, size_t dst_len);

    //------------------------------------------------------------------
    /// Resolves the user name at the beginning of \a src_path, and writes the output
    /// to \a dst_path.  Note, \a src_path can contain other path components after the
    /// user name, they will be copied over, and if the path doesn't start with "~" it
    /// will also be copied over to \a dst_path.
    ///
    /// @param[in] src_path
    ///     Input path to be resolved.
    ///
    /// @param[in] dst_path
    ///     Buffer to store the resolved path.
    ///
    /// @param[in] dst_len 
    ///     Size of the buffer pointed to by dst_path.
    ///
    /// @result 
    ///     The number of characters required to write the resolved path, or 0 if
    ///     the user name could not be found.  If the
    ///     resolved path doesn't fit in dst_len, dst_len-1 characters will
    ///     be written to \a dst_path, but the actual required length will still be returned.
    //------------------------------------------------------------------
    static size_t
    ResolveUsername (const char *src_path, char *dst_path, size_t dst_len);
    
    static size_t
    ResolvePartialUsername (const char *partial_name, StringList &matches);

    enum EnumerateDirectoryResult
    {
        eEnumerateDirectoryResultNext,  // Enumerate next entry in the current directory
        eEnumerateDirectoryResultEnter, // Recurse into the current entry if it is a directory or symlink, or next if not
        eEnumerateDirectoryResultExit,  // Exit from the current directory at the current level.
        eEnumerateDirectoryResultQuit   // Stop directory enumerations at any level
    };

    typedef EnumerateDirectoryResult (*EnumerateDirectoryCallbackType) (void *baton,
                                                                        FileType file_type,
                                                                        const FileSpec &spec
);

    static EnumerateDirectoryResult
    EnumerateDirectory (const char *dir_path,
                        bool find_directories,
                        bool find_files,
                        bool find_other,
                        EnumerateDirectoryCallbackType callback,
                        void *callback_baton);

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    ConstString m_directory;    ///< The uniqued directory path
    ConstString m_filename;     ///< The uniqued filename path
    mutable bool m_is_resolved; ///< True if this path has been resolved.
};

//----------------------------------------------------------------------
/// Dump a FileSpec object to a stream
//----------------------------------------------------------------------
Stream& operator << (Stream& s, const FileSpec& f);

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif  // liblldb_FileSpec_h_
