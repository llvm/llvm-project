#pragma once

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>

size_t getFileSize (FILE *input);

enum ERRORS {
    OK,
    CORRUPTED_ELF
};