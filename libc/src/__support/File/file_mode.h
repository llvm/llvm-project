
#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_FILE_MODE_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_FILE_MODE_H
#include <cstdint>

namespace LIBC_NAMESPACE_DECL {

// FileMode class handles everything regarding the mode of the file, be it's
// opening mode or content type.
class FileMode {
public:
  // FileMode constructor accepts the mode string as an argument.
  // It performs validation against several rules and records the `file_mode`
  // property to the specific mode.
  explicit FileMode(const char *mode) : file_mode_(0) {
    // First character in |mode| should be 'a', 'r' or 'w'.
    if (*mode != 'a' && *mode != 'r' && *mode != 'w')
      return;

    // There should be exactly one main mode ('a', 'r' or 'w') character.
    // If there are more than one main mode characters listed, then
    // we will consider |mode| as incorrect and return 0;
    int main_mode_count = 0;

    for (; *mode != '\0'; ++mode) {
      switch (*mode) {
      case 'r':
        file_mode_ |= static_cast<Mode>(OpenMode::READ);
        ++main_mode_count;
        break;
      case 'w':
        file_mode_ |= static_cast<Mode>(OpenMode::WRITE);
        ++main_mode_count;
        break;
      case '+':
        file_mode_ |= static_cast<Mode>(OpenMode::PLUS);
        break;
      case 'b':
        file_mode_ |= static_cast<Mode>(ContentType::BINARY);
        break;
      case 'a':
        file_mode_ |= static_cast<Mode>(OpenMode::APPEND);
        ++main_mode_count;
        break;
      case 'x':
        file_mode_ |= static_cast<Mode>(CreateType::EXCLUSIVE);
        break;
      default:
        file_mode_ = 0;
      }
    }

    if (main_mode_count != 1)
      file_mode_ = 0;
  }

  // helper function to show if file allows writing
  bool write_allowed() const {
    return (file_mode_ & static_cast<Mode>(OpenMode::WRITE)) != 0;
  }

  // helper function to show if file allows reading
  bool read_allowed() const {
    return (file_mode_ & static_cast<Mode>(OpenMode::READ)) != 0;
  }

  // helper function to show if file allows appending
  bool append_allowed() const {
    return (file_mode_ & static_cast<Mode>(OpenMode::APPEND)) != 0;
  }

  // helper function to denote if the file is in binary format.
  bool is_binary_format() const {
    return (file_mode_ & static_cast<Mode>(ContentType::BINARY)) != 0;
  }

  // '+' means update is allowed
  // TODO: ask michael if I need to give it a better name like "update_allowed"
  // or just continue with the old convention.
  bool is_plus() const {
    return (file_mode_ & static_cast<Mode>(OpenMode::PLUS)) != 0;
  }

  // checks if a file was created for writing
  bool is_exclusive_create() const {
    return (file_mode_ & static_cast<Mode>(CreateType::EXCLUSIVE)) != 0;
  }

private:
  // Mode is a generic or abstract mode bit for all kinds of modes
  // (open-mode, 'content-mode', 'create-modes')
  using Mode = uint32_t;

  // Denotes the mode of the file.
  //
  // The three different types of flags below are to be used with '|' operator.
  // Their values correspond to mutually exclusive bits in a 32-bit unsigned
  // integer value. A flag set can include both READ and WRITE if the file
  // is opened in update mode (ie. if the file was opened with a '+' the mode
  // string.)
  enum class OpenMode : Mode {
    READ = 0x1,
    WRITE = 0x2,
    APPEND = 0x4,
    PLUS = 0x8,
  };

  // Denotes a file opened in binary mode (which is specified by including
  // the 'b' character in teh mode string.)
  enum class ContentType : Mode {
    BINARY = 0x10,
  };

  // Denotes a file to be created for writing.
  enum class CreateType : Mode {
    EXCLUSIVE = 0x100,
  };

  // This property tracks the mode for the particular file instance (i.e
  // currently opened file)
  int file_mode_;
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_FILE_MODE_H
