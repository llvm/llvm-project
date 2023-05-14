#include "lib.hpp"

size_t getFileSize (FILE *input) {
    assert (input);

    int numberOfSymbols = 0;

    fseek (input, 0, SEEK_END);

    numberOfSymbols = ftell (input);

    fseek (input, 0, SEEK_SET);

    return numberOfSymbols;
}
