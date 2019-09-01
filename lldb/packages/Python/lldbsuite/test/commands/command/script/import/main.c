#include <stdio.h>

int main(int argc, char const *argv[]) {
    printf("Hello world.\n"); // Set break point at this line.
    if (argc == 1)
        return 0;

    // Waiting to be attached by the debugger, otherwise.
    char line[100];
    while (fgets(line, sizeof(line), stdin)) { // Waiting to be attached...
        printf("input line=>%s\n", line);
    }

    printf("Exiting now\n");
}
