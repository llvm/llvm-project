
#ifndef IPC_MANAGER_COMPILER_HPP
#define IPC_MANAGER_COMPILER_HPP

#include "clang/IPC2978/Manager.hpp"

#include <memory>
#include <unordered_map>

struct CompilerTest;
struct BuildSystemTest;
namespace P2978
{

enum class FileType : uint8_t
{
    MODULE,
    HEADER_UNIT,
    HEADER_FILE
};

struct Response
{
    // Borrows the manager's message storage; keep the manager alive while using this path.
    std::string_view filePath;
    // Empty for a textual header. BMI views remain mapped until the compiler process exits.
    std::string_view bmiContents;
    FileType type;
    bool isSystem;
};

// One dependency session per compiler process. Each request may populate several cached responses.
class IPCManagerCompiler : Manager
{
    friend struct ::CompilerTest;
    friend struct ::BuildSystemTest;

    Result<std::string_view> readInternal(char (&buffer)[4096]) const;
    // Send already-framed bytes through the compiler's pipe.
    Result<void> writeInternal(std::string_view buffer) const;

    // Decode a BMI path, reusing its mapped contents if an earlier response supplied it.
    Result<Response> readBMIResponse(std::string_view message, uint64_t &bytesRead, FileType type,
                                     bool isSystem = true);
    Result<void> readLogicalNames(std::string_view message, uint64_t &bytesRead, const Response &response);

    // Resolve a cache miss and retain the requested entry and all accompanying dependencies.
    [[nodiscard]] Result<void> receiveBTCModule(const CTBModule &moduleName);
    [[nodiscard]] Result<void> receiveBTCNonModule(const CTBNonModule &nonModule);

    // Logical-name keys and response paths borrow the retained message or mock-file storage below.
    std::unordered_map<std::string_view, Response> responses;

    // Paths borrow retained message or mock-file storage, just like responses.
    // Each path is mapped once per session. Erasing this cache would not unmap its views.
    std::unordered_map<std::string_view, std::string_view> bmiContentsByPath;

    // Initialized once so loading another mock cannot invalidate existing string views.
    std::string scanCacheFileData;
    bool isMocking = false;

    // Successful mappings belong to the process, not to the manager or the returned string_view.
    static Result<std::string_view> mapBMIFile(std::string_view filePath);
    // filePath must borrow retained message or mock-file storage.
    // Map the completed file only on the first request for its path.
    Result<std::string_view> loadBMIContents(std::string_view filePath);

    // Separate allocations keep string addresses stable as subsequent messages arrive.
    mutable std::vector<std::unique_ptr<std::string>> allocations;

  public:
    // Retained for Clang's command-line reproduction after successful mock initialization.
    std::string mockFilePath;

    // Load mock dependencies once on a fresh manager. A failed attempt also disables live IPC.
    Result<void> readEntriesFromFile(std::string_view filePath);

    // Look up a BMI already received from the build system; this does not map files or send requests.
    // Use the same normalized path supplied in the response (lowercase on Windows).
    [[nodiscard]] Result<std::string_view> findBMIContents(std::string_view filePath) const;

    // A textual-header request may resolve to a header unit for include translation. Other kinds must match.
    // Cache misses use one request/reply exchange; mock sessions report a missing entry instead.
    [[nodiscard]] Result<Response> findResponse(std::string_view logicalName, FileType type);
};

inline IPCManagerCompiler *managerCompiler;
} // namespace P2978
#endif // IPC_MANAGER_COMPILER_HPP
