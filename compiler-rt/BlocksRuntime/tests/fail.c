//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

/*
 *  fail.c
 *  testObjects
 *
 *  Created by Blaine Garst on 9/16/08.
 *
 */
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>


bool readfile(char *buffer, const char *from) {
    int fd = open(from, 0);
    if (fd < 0) return false;
    int count = read(fd, buffer, 512);
    if (count < 0) return false;
    buffer[count] = 0; // zap newline
    return true;
}

// basic idea, take compiler args, run compiler, and verify that expected failure matches any existing one

int main(int argc, char *argv[]) {
    if (argc == 1) return 0;
    char *copy[argc+1];   // make a copy
    // find and strip off -e "errorfile"
    char *errorfile = NULL;
    int counter = 0, i = 0;
    for (i = 1; i < argc; ++i) {    // skip 0 arg which is "fail"
        if (!strncmp(argv[i], "-e", 2)) {
            errorfile = argv[++i];
        }
        else {
            copy[counter++] = argv[i];
        }
    }
    copy[counter] = NULL;
    pid_t child = fork();
    char buffer[512];
    if (child == 0) {
        // in child
        sprintf(buffer, "/tmp/errorfile_%d", getpid());
        close(1);
        int fd = creat(buffer, 0777);
        if (fd != 1) {
            fprintf(stderr, "didn't open custom error file %s as 1, got %d\n", buffer, fd);
            exit(1);
        }
        close(2);
        dup(1);
        int result = execv(copy[0], copy);
        exit(10);
    }
    if (child < 0) {
        printf("fork failed\n");
        exit(1);
    }
    int status = 0;
    pid_t deadchild = wait(&status);
    if (deadchild != child) {
        printf("wait got %d instead of %d\n", deadchild, child);
        exit(1);
    }
    if (WEXITSTATUS(status) == 0) {
        printf("compiler exited normally, not good under these circumstances\n");
        exit(1);
    }
    //printf("exit status of child %d was %d\n", child, WEXITSTATUS(status));
    sprintf(buffer, "/tmp/errorfile_%d", child);
    if (errorfile) {
        //printf("ignoring error file: %s\n", errorfile);
        char desired[512];
        char got[512];
        bool gotErrorFile = readfile(desired, errorfile);
        bool gotOutput = readfile(got, buffer);
        if (!gotErrorFile && gotOutput) {
            printf("didn't read errorfile %s, it should have something from\n*****\n%s\n*****\nin it.\n",
                errorfile, got);
            exit(1);
        }
        else if (gotErrorFile && gotOutput) {
            char *where = strstr(got, desired);
            if (!where) {
                printf("didn't find contents of %s in %s\n", errorfile, buffer);
                exit(1);
            }
        }
        else {
            printf("errorfile %s and output %s inconsistent\n", errorfile, buffer);
            exit(1);
        }
    }
    unlink(buffer);
    printf("success\n");
    exit(0);
}
        
