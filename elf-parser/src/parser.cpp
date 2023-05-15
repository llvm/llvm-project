#include "parser.hpp"

uint8_t *createBuffer (const char *inputFileName) {
    assert (inputFileName);

    FILE *elfFile = fopen(inputFileName, "r");
    if (!elfFile) {
        std::cout << "Error: cannot open " << inputFileName << "\n";
        return nullptr;
    }

    size_t fileSize = getFileSize (elfFile);

    uint8_t *binary = (uint8_t *)calloc (fileSize + 1, sizeof (uint8_t));
    if (!binary) {
        std::cout << "Unable to allocate memory!\n";
        return nullptr;
    }

    size_t numberOfReadBytes = fread (binary, sizeof (uint8_t), fileSize, elfFile);
    if (numberOfReadBytes != fileSize) {
        std::cout << "Incorrect file reading occured!\n";
        return nullptr;
    }

    return binary;
}

Elf64_Sym_Arr *getSymbols (Elf64_Ehdr *elfHeader) {
    assert (elfHeader);

    uint8_t *binary = (uint8_t *)elfHeader;

    Elf64_Shdr *sections = (Elf64_Shdr *)(binary + elfHeader->e_shoff);
    Elf64_Shdr *shStrSect = &sections[elfHeader->e_shstrndx];

    uint8_t *shStrSectPtr = binary + shStrSect->sh_offset;

    int symTabIndex = -1;
    int strTabIndex = -1;

    for (int index = 0; index < elfHeader->e_shnum; index++) {
        if (strTabIndex != -1 && symTabIndex != -1) {
            break;
        }

        int strOffset = sections[index].sh_name;

        if (strcmp (".symtab", (char *)(shStrSectPtr + strOffset)) == 0) {
            symTabIndex = index;
        }

        if (strcmp (".strtab", (char *)(shStrSectPtr + strOffset)) == 0) {
            strTabIndex = index;
        }
    }

    if (symTabIndex == -1 || strTabIndex == -1) {
        std::cout << "File is corrupted!\n";
        return nullptr;
    }

    Elf64_Shdr *symTabSect = &sections[symTabIndex];
    size_t numberOfSymbols = symTabSect->sh_size / symTabSect->sh_entsize;
    Elf64_Sym *symbols = (Elf64_Sym *)(binary + symTabSect->sh_offset);

    Elf64_Sym_Arr *symArray = (Elf64_Sym_Arr *)calloc (1, sizeof (Elf64_Sym_Arr));
    if (!symArray) {
        std::cout << "Unable to allocate memory!\n";
        return nullptr;
    }

    symArray->symbols = (Elf64_Sym_W_Name *)calloc (numberOfSymbols, sizeof (Elf64_Sym_W_Name));
    if (!symArray->symbols) {
        std::cout << "Unable to allocate memory!\n";
        return nullptr;
    }

    symArray->size = numberOfSymbols;

    // Check if all works correctly.
    Elf64_Shdr *strTabSect = &sections[strTabIndex];
    char *strTabPtr = (char *)(binary + strTabSect->sh_offset);

    for (int i = 0; i < numberOfSymbols; i++) {
        // std::cout << symbols[i].st_value << "       " << strTabPtr + symbols[i].st_name << '\n';
        symArray->symbols[i].symbol = &symbols[i];
        symArray->symbols[i].symName = strTabPtr + symbols[i].st_name;
    }


    return symArray;
}

int symbolComp (const void *symbol1, const void *symbol2) {
    assert (symbol1);
    assert (symbol2);

    Elf64_Sym_W_Name *sym1 = (Elf64_Sym_W_Name *)symbol1;
    Elf64_Sym_W_Name *sym2 = (Elf64_Sym_W_Name *)symbol2;

    return sym1->symbol->st_value - sym2->symbol->st_value;
}

void printSymbolsValues (Elf64_Sym_Arr *symArr) {
    assert (symArr);

    for (size_t index = 0; index < symArr->size; index++) {
        std::cout << "Value: " << symArr->symbols[index].symbol->st_value << "      " << symArr->symbols[index].symName << '\n';
    }

    std::cout << '\n';
}

Elf64_Sym_W_Name *findSymbolByAddress (Elf64_Sym_Arr *symArr, size_t address) {
    assert (symArr);

    size_t leftIndex = 0;
    size_t rightIndex = symArr->size - 1;

    while (leftIndex < rightIndex) {
        size_t midIndex = leftIndex + (rightIndex - leftIndex) / 2;

        if (address == symArr->symbols[midIndex].symbol->st_value) {
            return &symArr->symbols[midIndex];
        }
        else if (address < symArr->symbols[midIndex].symbol->st_value) {
            rightIndex = midIndex;
        }
        else {
            leftIndex = midIndex;
        }
    }

    return &symArr->symbols[leftIndex];
}

