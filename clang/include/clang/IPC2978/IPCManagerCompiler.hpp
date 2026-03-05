
#ifndef IPC_MANAGER_COMPILER_HPP
#define IPC_MANAGER_COMPILER_HPP

#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/expected.hpp"

struct CompilerTest;
struct BuildSystemTest;
namespace P2978
{

inline std::vector<std::string *> allocations;

enum class FileType : uint8_t
{
    MODULE,
    HEADER_UNIT,
    HEADER_FILE
};

struct Response
{
    std::string_view filePath;
    // if type == HEADER_FILE, then fileSize has no meaning
    Mapping mapping;
    FileType type;
    bool isSystem;
    Response(std::string_view filePath_, const Mapping &mapping_, FileType type_, bool isSystem_);
};

// IPC Manager Compiler
class IPCManagerCompiler : Manager
{
    friend struct ::CompilerTest;
    friend struct ::BuildSystemTest;

    tl::expected<std::string_view, std::string> readInternal(char (&buffer)[4096]) const;
    tl::expected<void, std::string> writeInternal(std::string_view buffer) const override;

    struct BMIFileMapping
    {
        BMIFile file;
        Mapping mapping;
    };

    tl::expected<BMIFileMapping, std::string> readProcessMappingOfBMIFile(std::string_view message,
                                                                          uint32_t &bytesRead);
    static tl::expected<ModuleDep, std::string> readModuleDep(std::string_view message, uint32_t &bytesRead);
    static tl::expected<HuDep, std::string> readHuDep(std::string_view message, uint32_t &bytesRead);
    static tl::expected<HeaderFile, std::string> readHeaderFile(std::string_view message, uint32_t &bytesRead);
    tl::expected<void, std::string> readLogicalNames(std::string_view message, uint32_t &bytesRead,
                                                     const BMIFileMapping &mapping, FileType type, bool isSystem);

    // Called by sendCTBLastMessage. Build-system will send this after it has created the BMI file-mapping.
    [[nodiscard]] tl::expected<void, std::string> receiveBTCLastMessage() const;
    // This function is called by findResponse if it did not find the module in the IPCManagerCompiler::responses cache.
    [[nodiscard]] tl::expected<void, std::string> receiveBTCModule(const CTBModule &moduleName);
    // This function is called by findResponse if it did not find the header-unit or header-file in the
    // IPCManagerCompiler::responses cache.
    [[nodiscard]] tl::expected<void, std::string> receiveBTCNonModule(const CTBNonModule &nonModule);

    // Internal cache for the possible future requests.
    std::unordered_map<std::string_view, Response> responses;

    //  Compiler can use this function to read the BMI file. BMI should be read using this function to conserve memory.
    static tl::expected<Mapping, std::string> readSharedMemoryBMIFile(const BMIFile &file);

    [[nodiscard]] tl::expected<void, std::string> sendCTBLastMessage(uint32_t fileSize) const;

  public:
    // Compiler process can use this function to close the BMI file-mapping to reduce references to shared memory file.
    // Not needed as it will be cleared at process exit.
    static tl::expected<void, std::string> closeBMIFileMapping(const Mapping &processMappingOfBMIFile);

    // Cache mapping between the file-path and bmi-file-mapping. Only to be queried by the compiler. Passed path must be
    // lexically normal and lower-case on Windows.
    std::unordered_map<std::string, Mapping> filePathProcessMapping;

    // For FileType::HEADER_FILE, it can return FileType::HEADER_UNIT, otherwise it will return the request
    // response. Either it will return from the cache or it will fetch it from the build-system
    [[nodiscard]] tl::expected<Response, std::string> findResponse(std::string_view logicalName, FileType type);

    // This function should be called only if the compilation succeeded
    [[nodiscard]] tl::expected<void, std::string> sendCTBLastMessage(const std::string &bmiFile,
                                                                     const std::string &filePath) const;
};

inline IPCManagerCompiler *managerCompiler;
} // namespace P2978
#endif // IPC_MANAGER_COMPILER_HPP
