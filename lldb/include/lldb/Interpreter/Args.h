//===-- Args.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Command_h_
#define liblldb_Command_h_

// C Includes
// C++ Includes
#include <list>
#include <string>
#include <utility>
#include <vector>

// Other libraries and framework includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
// Project includes
#include "lldb/Utility/Error.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

struct Option;

typedef std::vector<std::tuple<std::string, int, std::string>> OptionArgVector;
typedef std::shared_ptr<OptionArgVector> OptionArgVectorSP;

struct OptionArgElement {
  enum { eUnrecognizedArg = -1, eBareDash = -2, eBareDoubleDash = -3 };

  OptionArgElement(int defs_index, int pos, int arg_pos)
      : opt_defs_index(defs_index), opt_pos(pos), opt_arg_pos(arg_pos) {}

  int opt_defs_index;
  int opt_pos;
  int opt_arg_pos;
};

typedef std::vector<OptionArgElement> OptionElementVector;

//----------------------------------------------------------------------
/// @class Args Args.h "lldb/Interpreter/Args.h"
/// @brief A command line argument class.
///
/// The Args class is designed to be fed a command line. The
/// command line is copied into an internal buffer and then split up
/// into arguments. Arguments are space delimited if there are no quotes
/// (single, double, or backtick quotes) surrounding the argument. Spaces
/// can be escaped using a \ character to avoid having to surround an
/// argument that contains a space with quotes.
//----------------------------------------------------------------------
class Args {
public:
  struct ArgEntry {
  private:
    friend class Args;
    std::unique_ptr<char[]> ptr;

    char *data() { return ptr.get(); }

  public:
    ArgEntry() = default;
    ArgEntry(llvm::StringRef str, char quote);

    llvm::StringRef ref;
    char quote;
    const char *c_str() const { return ptr.get(); }
  };

  //------------------------------------------------------------------
  /// Construct with an option command string.
  ///
  /// @param[in] command
  ///     A NULL terminated command that will be copied and split up
  ///     into arguments.
  ///
  /// @see Args::SetCommandString(llvm::StringRef)
  //------------------------------------------------------------------
  Args(llvm::StringRef command = llvm::StringRef());

  Args(const Args &rhs);

  Args &operator=(const Args &rhs);

  //------------------------------------------------------------------
  /// Destructor.
  //------------------------------------------------------------------
  ~Args();

  //------------------------------------------------------------------
  /// Dump all entries to the stream \a s using label \a label_name.
  ///
  /// If label_name is nullptr, the dump operation is skipped.
  ///
  /// @param[in] s
  ///     The stream to which to dump all arguments in the argument
  ///     vector.
  /// @param[in] label_name
  ///     The label_name to use as the label printed for each
  ///     entry of the args like so:
  ///       {label_name}[{index}]={value}
  //------------------------------------------------------------------
  void Dump(Stream &s, const char *label_name = "argv") const;

  //------------------------------------------------------------------
  /// Sets the command string contained by this object.
  ///
  /// The command string will be copied and split up into arguments
  /// that can be accessed via the accessor functions.
  ///
  /// @param[in] command
  ///     A command StringRef that will be copied and split up
  ///     into arguments.
  ///
  /// @see Args::GetArgumentCount() const
  /// @see Args::GetArgumentAtIndex (size_t) const
  /// @see Args::GetArgumentVector ()
  /// @see Args::Shift ()
  /// @see Args::Unshift (const char *)
  //------------------------------------------------------------------
  void SetCommandString(llvm::StringRef command);

  bool GetCommandString(std::string &command) const;

  bool GetQuotedCommandString(std::string &command) const;

  //------------------------------------------------------------------
  /// Gets the number of arguments left in this command object.
  ///
  /// @return
  ///     The number or arguments in this object.
  //------------------------------------------------------------------
  size_t GetArgumentCount() const;
  bool empty() const { return GetArgumentCount() == 0; }

  //------------------------------------------------------------------
  /// Gets the NULL terminated C string argument pointer for the
  /// argument at index \a idx.
  ///
  /// @return
  ///     The NULL terminated C string argument pointer if \a idx is a
  ///     valid argument index, NULL otherwise.
  //------------------------------------------------------------------
  const char *GetArgumentAtIndex(size_t idx) const;

  llvm::ArrayRef<ArgEntry> entries() const { return m_entries; }
  char GetArgumentQuoteCharAtIndex(size_t idx) const;

  std::vector<ArgEntry>::const_iterator begin() const {
    return m_entries.begin();
  }
  std::vector<ArgEntry>::const_iterator end() const { return m_entries.end(); }

  size_t size() const { return GetArgumentCount(); }
  const ArgEntry &operator[](size_t n) const { return m_entries[n]; }

  //------------------------------------------------------------------
  /// Gets the argument vector.
  ///
  /// The value returned by this function can be used by any function
  /// that takes and vector. The return value is just like \a argv
  /// in the standard C entry point function:
  ///     \code
  ///         int main (int argc, const char **argv);
  ///     \endcode
  ///
  /// @return
  ///     An array of NULL terminated C string argument pointers that
  ///     also has a terminating NULL C string pointer
  //------------------------------------------------------------------
  char **GetArgumentVector();

  //------------------------------------------------------------------
  /// Gets the argument vector.
  ///
  /// The value returned by this function can be used by any function
  /// that takes and vector. The return value is just like \a argv
  /// in the standard C entry point function:
  ///     \code
  ///         int main (int argc, const char **argv);
  ///     \endcode
  ///
  /// @return
  ///     An array of NULL terminate C string argument pointers that
  ///     also has a terminating NULL C string pointer
  //------------------------------------------------------------------
  const char **GetConstArgumentVector() const;

  //------------------------------------------------------------------
  /// Gets the argument as an ArrayRef. Note that the return value does *not*
  /// have a nullptr const char * at the end, as the size of the list is
  /// embedded in the ArrayRef object.
  //------------------------------------------------------------------
  llvm::ArrayRef<const char *> GetArgumentArrayRef() const {
    return llvm::makeArrayRef(m_argv).drop_back();
  }

  //------------------------------------------------------------------
  /// Appends a new argument to the end of the list argument list.
  ///
  /// @param[in] arg_cstr
  ///     The new argument as a NULL terminated C string.
  ///
  /// @param[in] quote_char
  ///     If the argument was originally quoted, put in the quote char here.
  //------------------------------------------------------------------
  void AppendArgument(llvm::StringRef arg_str, char quote_char = '\0');

  void AppendArguments(const Args &rhs);

  void AppendArguments(const char **argv);

  //------------------------------------------------------------------
  /// Insert the argument value at index \a idx to \a arg_cstr.
  ///
  /// @param[in] idx
  ///     The index of where to insert the argument.
  ///
  /// @param[in] arg_cstr
  ///     The new argument as a NULL terminated C string.
  ///
  /// @param[in] quote_char
  ///     If the argument was originally quoted, put in the quote char here.
  ///
  /// @return
  ///     The NULL terminated C string of the copy of \a arg_cstr.
  //------------------------------------------------------------------
  void InsertArgumentAtIndex(size_t idx, llvm::StringRef arg_str,
                             char quote_char = '\0');

  //------------------------------------------------------------------
  /// Replaces the argument value at index \a idx to \a arg_cstr
  /// if \a idx is a valid argument index.
  ///
  /// @param[in] idx
  ///     The index of the argument that will have its value replaced.
  ///
  /// @param[in] arg_cstr
  ///     The new argument as a NULL terminated C string.
  ///
  /// @param[in] quote_char
  ///     If the argument was originally quoted, put in the quote char here.
  //------------------------------------------------------------------
  void ReplaceArgumentAtIndex(size_t idx, llvm::StringRef arg_str,
                              char quote_char = '\0');

  //------------------------------------------------------------------
  /// Deletes the argument value at index
  /// if \a idx is a valid argument index.
  ///
  /// @param[in] idx
  ///     The index of the argument that will have its value replaced.
  ///
  //------------------------------------------------------------------
  void DeleteArgumentAtIndex(size_t idx);

  //------------------------------------------------------------------
  /// Sets the argument vector value, optionally copying all
  /// arguments into an internal buffer.
  ///
  /// Sets the arguments to match those found in \a argv. All argument
  /// strings will be copied into an internal buffers.
  //
  //  FIXME: Handle the quote character somehow.
  //------------------------------------------------------------------
  void SetArguments(size_t argc, const char **argv);

  void SetArguments(const char **argv);

  //------------------------------------------------------------------
  /// Shifts the first argument C string value of the array off the
  /// argument array.
  ///
  /// The string value will be freed, so a copy of the string should
  /// be made by calling Args::GetArgumentAtIndex (size_t) const
  /// first and copying the returned value before calling
  /// Args::Shift().
  ///
  /// @see Args::GetArgumentAtIndex (size_t) const
  //------------------------------------------------------------------
  void Shift();

  //------------------------------------------------------------------
  /// Inserts a class owned copy of \a arg_cstr at the beginning of
  /// the argument vector.
  ///
  /// A copy \a arg_cstr will be made.
  ///
  /// @param[in] arg_cstr
  ///     The argument to push on the front of the argument stack.
  ///
  /// @param[in] quote_char
  ///     If the argument was originally quoted, put in the quote char here.
  //------------------------------------------------------------------
  void Unshift(llvm::StringRef arg_str, char quote_char = '\0');

  //------------------------------------------------------------------
  /// Parse the arguments in the contained arguments.
  ///
  /// The arguments that are consumed by the argument parsing process
  /// will be removed from the argument vector. The arguments that
  /// get processed start at the second argument. The first argument
  /// is assumed to be the command and will not be touched.
  ///
  /// param[in] platform_sp
  ///   The platform used for option validation.  This is necessary
  ///   because an empty execution_context is not enough to get us
  ///   to a reasonable platform.  If the platform isn't given,
  ///   we'll try to get it from the execution context.  If we can't
  ///   get it from the execution context, we'll skip validation.
  ///
  /// param[in] require_validation
  ///   When true, it will fail option parsing if validation could
  ///   not occur due to not having a platform.
  ///
  /// @see class Options
  //------------------------------------------------------------------
  Error ParseOptions(Options &options, ExecutionContext *execution_context,
                     lldb::PlatformSP platform_sp, bool require_validation);

  bool IsPositionalArgument(const char *arg);

  // The following works almost identically to ParseOptions, except that no
  // option is required to have arguments, and it builds up the
  // option_arg_vector as it parses the options.

  std::string ParseAliasOptions(Options &options, CommandReturnObject &result,
                                OptionArgVector *option_arg_vector,
                                llvm::StringRef raw_input_line);

  void ParseArgsForCompletion(Options &options,
                              OptionElementVector &option_element_vector,
                              uint32_t cursor_index);

  //------------------------------------------------------------------
  // Clear the arguments.
  //
  // For re-setting or blanking out the list of arguments.
  //------------------------------------------------------------------
  void Clear();

  static const char *StripSpaces(std::string &s, bool leading = true,
                                 bool trailing = true,
                                 bool return_null_if_empty = true);

  static bool UInt64ValueIsValidForByteSize(uint64_t uval64,
                                            size_t total_byte_size) {
    if (total_byte_size > 8)
      return false;

    if (total_byte_size == 8)
      return true;

    const uint64_t max = ((uint64_t)1 << (uint64_t)(total_byte_size * 8)) - 1;
    return uval64 <= max;
  }

  static bool SInt64ValueIsValidForByteSize(int64_t sval64,
                                            size_t total_byte_size) {
    if (total_byte_size > 8)
      return false;

    if (total_byte_size == 8)
      return true;

    const int64_t max = ((int64_t)1 << (uint64_t)(total_byte_size * 8 - 1)) - 1;
    const int64_t min = ~(max);
    return min <= sval64 && sval64 <= max;
  }

  static lldb::addr_t StringToAddress(const ExecutionContext *exe_ctx,
                                      llvm::StringRef s,
                                      lldb::addr_t fail_value, Error *error);

  static bool StringToBoolean(llvm::StringRef s, bool fail_value,
                              bool *success_ptr);

  static char StringToChar(llvm::StringRef s, char fail_value,
                           bool *success_ptr);

  static int64_t StringToOptionEnum(llvm::StringRef s,
                                    OptionEnumValueElement *enum_values,
                                    int32_t fail_value, Error &error);

  static lldb::ScriptLanguage
  StringToScriptLanguage(llvm::StringRef s, lldb::ScriptLanguage fail_value,
                         bool *success_ptr);

  // TODO: Use StringRef
  static Error StringToFormat(const char *s, lldb::Format &format,
                              size_t *byte_size_ptr); // If non-NULL, then a
                                                      // byte size can precede
                                                      // the format character

  static lldb::Encoding
  StringToEncoding(llvm::StringRef s,
                   lldb::Encoding fail_value = lldb::eEncodingInvalid);

  static uint32_t StringToGenericRegister(llvm::StringRef s);

  static bool StringToVersion(llvm::StringRef string, uint32_t &major,
                              uint32_t &minor, uint32_t &update);

  static const char *GetShellSafeArgument(const FileSpec &shell,
                                          const char *unsafe_arg,
                                          std::string &safe_arg);

  // EncodeEscapeSequences will change the textual representation of common
  // escape sequences like "\n" (two characters) into a single '\n'. It does
  // this for all of the supported escaped sequences and for the \0ooo (octal)
  // and \xXX (hex). The resulting "dst" string will contain the character
  // versions of all supported escape sequences. The common supported escape
  // sequences are: "\a", "\b", "\f", "\n", "\r", "\t", "\v", "\'", "\"", "\\".

  static void EncodeEscapeSequences(const char *src, std::string &dst);

  // ExpandEscapeSequences will change a string of possibly non-printable
  // characters and expand them into text. So '\n' will turn into two characters
  // like "\n" which is suitable for human reading. When a character is not
  // printable and isn't one of the common in escape sequences listed in the
  // help for EncodeEscapeSequences, then it will be encoded as octal. Printable
  // characters are left alone.
  static void ExpandEscapedCharacters(const char *src, std::string &dst);

  static std::string EscapeLLDBCommandArgument(const std::string &arg,
                                               char quote_char);

  //------------------------------------------------------------------
  /// Add or replace an environment variable with the given value.
  ///
  /// This command adds the environment variable if it is not already
  /// present using the given value.  If the environment variable is
  /// already in the list, it replaces the first such occurrence
  /// with the new value.
  //------------------------------------------------------------------
  void AddOrReplaceEnvironmentVariable(llvm::StringRef env_var_name,
                                       llvm::StringRef new_value);

  /// Return whether a given environment variable exists.
  ///
  /// This command treats Args like a list of environment variables,
  /// as used in ProcessLaunchInfo.  It treats each argument as
  /// an {env_var_name}={value} or an {env_var_name} entry.
  ///
  /// @param[in] env_var_name
  ///     Specifies the name of the environment variable to check.
  ///
  /// @param[out] argument_index
  ///     If non-null, then when the environment variable is found,
  ///     the index of the argument position will be returned in
  ///     the size_t pointed to by this argument.
  ///
  /// @return
  ///     true if the specified env var name exists in the list in
  ///     either of the above-mentioned formats; otherwise, false.
  //------------------------------------------------------------------
  bool ContainsEnvironmentVariable(llvm::StringRef env_var_name,
                                   size_t *argument_index = nullptr) const;

private:
  size_t FindArgumentIndexForOption(Option *long_options,
                                    int long_options_index) const;

  std::vector<ArgEntry> m_entries;
  std::vector<char *> m_argv;

  void UpdateArgsAfterOptionParsing();
};

} // namespace lldb_private

#endif // liblldb_Command_h_
