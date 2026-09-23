#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <cstdint>
#include <string_view>
#include <vector>

namespace P2978
{
// CTB: compiler to build system. BTC: build system to compiler.
// Strings use a uint32_t byte length followed by their bytes. Paths add a NUL
// outside that length. Vectors use a uint32_t element count followed by elements.
// Integers are native-endian; bools and request types occupy one byte.
// Fields are serialized in declaration order, except for BTCNonModule's optional tail.

// A request begins with this tag. Each compiler waits for its response before sending another request.
enum class CTB : uint8_t
{
    MODULE = 0,
    NON_MODULE = 1,
};

// Resolve a named module or module partition.
struct CTBModule
{
    std::string_view moduleName;
};

// Resolve an include or a header-unit import. An include may resolve to a header unit.
struct CTBNonModule
{
    bool isHeaderUnit = false;
    std::string_view logicalName;
};

// Responses need no type tag: the outstanding request determines which layout to read.
// A BMI must be complete before publication and remain available to its consumers.
struct ModuleDep
{
    bool isHeaderUnit = false;
    std::string_view filePath;
    // Classify the dependency as system input for compiler diagnostics.
    bool isSystem = true;
    // A module has one name. A composed header unit may provide several include-name aliases.
    std::vector<std::string_view> logicalNames;
};

// Reply to CTBModule, including dependencies needed to load the requested BMI.
struct BTCModule
{
    std::string_view filePath;
    bool isSystem = true;
    // Omit dependencies already supplied to this compiler; their cached responses remain valid.
    std::vector<ModuleDep> modDeps;
};

struct HuDep
{
    std::string_view filePath;
    // Classify the header unit as system input for compiler diagnostics.
    bool isSystem = true;
    // Include names that resolve to this same header-unit BMI.
    std::vector<std::string_view> logicalNames;
};

struct HeaderFile
{
    std::string_view logicalName;
    std::string_view filePath;
    bool isSystem = true;
};

// Reply to CTBNonModule. Batch known headers and BMI dependencies to avoid further round trips.
struct BTCNonModule
{
    bool isHeaderUnit = false;
    bool isSystem = true;
    // Additional textual headers to cache, for example the headers composing the unit being built.
    std::vector<HeaderFile> headerFiles;
    std::string_view filePath;
    // The remaining fields are sent only for a header-unit response. Otherwise filePath names a textual header.
    // Additional aliases for the requested BMI; the request's logical name is cached implicitly.
    std::vector<std::string_view> logicalNames;
    std::vector<HuDep> huDeps;
};

} // namespace P2978
#endif // MESSAGES_HPP
