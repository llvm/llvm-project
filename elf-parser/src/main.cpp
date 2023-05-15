#include <iostream>
#include "parser.hpp"

int main (int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Error: missing file name!\n";
        return -1;
    }

    uint8_t *binary = createBuffer (argv[1]);
    if (!binary) {
        return -1;
    }

    Elf64_Ehdr *elfHeader = (Elf64_Ehdr *)binary;
    Elf64_Sym_Arr *symbolArr = getSymbols(elfHeader);
    if (!symbolArr) {
        return -1;
    }

    // printSymbolsValues (symbolArr);
    qsort ((void *)symbolArr->symbols, symbolArr->size, sizeof (Elf64_Sym_W_Name), symbolComp);
    // printSymbolsValues (symbolArr);

    size_t numberOfStrings = 0;
    char *addrs = (char *)createBuffer (argv[2], &numberOfStrings);
    if (!addrs) {
        return -1;
    }

    std::cout << symbolArr->size << '\n';

    std::cout << numberOfStrings << std::endl;

    std::map <std::pair<u_int64_t, u_int64_t>, int> funcHashTable; 
    char **strArray = new char *[numberOfStrings];
    initializeArrOfPointers (strArray, numberOfStrings, addrs);

    for (size_t i = 0; i < numberOfStrings; i++) {
        std::cout << strArray[i] << '\n';
    }

    fillHashMap (funcHashTable, strArray, numberOfStrings, symbolArr);
    dumpMapToFile (funcHashTable, symbolArr);

    return 0;
}