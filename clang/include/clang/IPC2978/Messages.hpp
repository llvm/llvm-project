#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <cstdint>
#include <signal.h>
#include <string_view>
#include <vector>

namespace P2978
{
// CTB --> Compiler to Build-System
// BTC --> Build-System to Compiler

// string_view is 4 bytes that hold the size of the char array, followed by the array.
// string_view representing the filePath is followed by null terminator.
// vector is 4 bytes that hold the size of the array, followed by the array.
// All fields are sent in declaration order, even if meaningless.

// Compiler to Build System
// This is the first byte of the compiler to build-system message.
enum class CTB : uint8_t
{
    MODULE = 0,
    NON_MODULE = 1,
    LAST_MESSAGE = 2,
};

// This is sent when the compiler needs a module.
struct CTBModule
{
    std::string_view moduleName;
};

// This is sent when the compiler needs something else than a module.
// isHeaderUnit is set when the compiler knows that it is a header-unit.
struct CTBNonModule
{
    bool isHeaderUnit = false;
    std::string_view logicalName;
};

// This is the last message sent by the compiler if the compiler
// has any exported BMI.
struct CTBLastMessage
{
    // This is communicated because the receiving process has no
    // way to learn the shared memory file size on both Windows
    // and Linux without a filesystem call.
    // Meaningless if the compilation does not produce BMI.
    uint32_t fileSize = UINT32_MAX;
};

// Build System to Compiler
// Unlike CTB, this is not written as the first byte
// since the compiler knows what message it will receive.
enum class BTC : uint8_t
{
    MODULE = 0,
    NON_MODULE = 1,
    LAST_MESSAGE = 2,
};

struct BMIFile
{
    std::string_view filePath;
    uint32_t fileSize = UINT32_MAX;
};

struct ModuleDep
{
    bool isHeaderUnit;
    BMIFile file;
    // whether header-unit / module belongs to system (ignore warnings).
    bool isSystem = true;
    // if isHeaderUnit == true, then the following might
    // contain more than one values, as header-unit can be
    // composed of multiple header-files. And if later,
    // any of the following logicalNames is included or
    // imported, this header-unit can be used instead.
    std::vector<std::string_view> logicalNames;
};

// Reply for CTBModule
struct BTCModule
{
    BMIFile requested;
    bool isSystem = true;
    std::vector<ModuleDep> modDeps;
};

struct HuDep
{
    BMIFile file;
    // whether header-unit / header-file belongs to system (ignore warnings).
    bool isSystem = true;
    // A header-unit can be composed of
    // multiple header-files. And if later,
    // any of the following logicalNames is included or
    // imported, this header-unit can be used instead.
    std::vector<std::string_view> logicalNames;
};

struct HeaderFile
{
    std::string_view logicalName;
    std::string_view filePath;
    bool isSystem = true;
};

// Reply for CTBNonModule
struct BTCNonModule
{
    bool isHeaderUnit = false;
    bool isSystem = true;
    // build-system might send the following on first request, if it knows that a
    // header-unit is being compiled that compose multiple header-files to reduce
    // the number of subsequent requests.
    std::vector<HeaderFile> headerFiles;
    std::string_view filePath;
    // if isHeaderUnit == false, the following are meaning-less and are not sent.
    // if isHeaderUnit == true, fileSize of the requested file.
    uint32_t fileSize;
    // A header-unit can be composed of
    // multiple header-files. And if later,
    // any of the following logicalNames is included or
    // imported, this header-unit can be used instead.
    std::vector<std::string_view> logicalNames;
    std::vector<HuDep> huDeps;
};

// Reply for CTBLastMessage if the compilation succeeded.
struct BTCLastMessage
{
};
} // namespace P2978
#endif // MESSAGES_HPP
