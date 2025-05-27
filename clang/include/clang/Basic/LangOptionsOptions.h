#ifndef LLVM_CLANG_INCLUDE_BASIC_LANGOPTIONSOPTIONS_H
#define LLVM_CLANG_INCLUDE_BASIC_LANGOPTIONSOPTIONS_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include <cstddef>
#include <string>
#include <unordered_map>

using namespace clang;

struct LangOptionsOption {
    // Completely ignores this option when it comes to 
    bool ignore_mismatch;
};

// Options that can be applied arbitrarily to any langopt
struct LangOptionsOptions {

#define LANGOPT(Name, Bits, Default, Description) LangOptionsOption Name;
#include "clang/Basic/LangOptions.def"

    LangOptionsOption* get(const std::string& OptName) {
        #define LANGOPT(Name, Bits, Default, Description) \
          {std::string(#Name), offsetof(LangOptionsOptions, Name)},
        static const std::unordered_map<std::string, size_t> Offsets = {
            #include "clang/Basic/LangOptions.def"
        };
        if (auto it = Offsets.find(OptName); it != Offsets.end()) {
            auto Offset = it->second;
            return reinterpret_cast<LangOptionsOption*>(reinterpret_cast<std::byte*>(this) + Offset);
        } else {
            return nullptr;
        }
    }
};

#endif // LLVM_CLANG_INCLUDE_BASIC_LANGOPTIONSOPTIONS_H