#pragma once

#include <elf.h>

#include "lib.hpp"

struct Elf64_Sym_W_Name {
    Elf64_Sym *symbol;
    char *symName;
};

struct Elf64_Sym_Arr {
    Elf64_Sym_W_Name *symbols;
    size_t size;
};

uint8_t *createBuffer (const char *inputFileName, size_t *numberOfStrings = nullptr);

void fillHashMap (std::map <std::pair<uint64_t, u_int64_t>, int> &funcHashTable, char **strArray, size_t numberOfStrings, Elf64_Sym_Arr *symArray);

Elf64_Sym_Arr *getSymbols (Elf64_Ehdr *elfHeader);

int symbolComp (const void *symbol1, const void *symbol2);

void printSymbolsValues (Elf64_Sym_Arr *symArr);

Elf64_Sym_W_Name *findSymbolByAddress (Elf64_Sym_Arr *symArr, size_t address);

void dumpMapToFile (std::map <std::pair<u_int64_t, u_int64_t>, int> &funcHashTable, Elf64_Sym_Arr *symArr);
bool isPIC(const char *inputFileName);
