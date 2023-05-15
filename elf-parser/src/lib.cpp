#include "lib.hpp"

size_t getFileSize (FILE *input) {
    assert (input);

    int numberOfSymbols = 0;

    fseek (input, 0, SEEK_END);

    numberOfSymbols = ftell (input);

    fseek (input, 0, SEEK_SET);

    return numberOfSymbols;
}

size_t getNumberOfStrings (FILE *input) {
    assert (input);

    int number = 0;

    char symbol = (char)getc (input);

    while (symbol != EOF) {
        if (symbol == '\n') {
            number++;
        } else if (symbol == '\0') {
            number++;
            break;
        }

        symbol = (char)getc (input);
    }

    fseek (input, 0, SEEK_SET);

    return number;
}

char *skipNonLetters (char *src) {
    assert (src);

    while (!isalpha((int)*src) && *src != '\0') {
        src++;
    }

    return src;
}

char *bufferAlloc (size_t *fileS, FILE *input) {
    assert (fileS);
    assert (input);

    char *src = (char*)calloc (*fileS, sizeof (char));
    if (src == nullptr) {
        return nullptr;
    }

    *fileS = fread (src, sizeof (char), *fileS, input);
    fseek (input, 0, SEEK_SET);

    return src;
}

void initializeArrOfPointers (char **ptrToStrArr, const size_t numbOfStrings, char *text) {
    for (size_t i = 0; i < numbOfStrings; i++) {
        ptrToStrArr[i] = text;

        while (*text != '\n' && *text != '\0') {
            text++;
        }

        if (*text == '\0') {
            break;
        }

        *text = '\0';
        text++;
    }
}



u_int64_t getHash (char *src) {
    assert (src);

    u_int64_t hash = 0;
    sscanf (src, "%lu", &hash);
    if (!hash) {
        printf ("Error: file is corrupted!\n");
    }

    return hash;
}
