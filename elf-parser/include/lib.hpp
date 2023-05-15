#pragma once

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <map>
#include <unordered_set>
#include <fstream>

size_t getFileSize (FILE *input);

size_t getNumberOfStrings (FILE *input);

char *bufferAlloc (size_t *fileS, FILE *input);

char *skipNonLetters (char *src);

void initializeArrOfPointers (char **ptrToStrArr, const size_t numbOfStrings, char *text);

u_int64_t getHash (char *src);

enum ERRORS {
    OK,
    CORRUPTED_ELF
};