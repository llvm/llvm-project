//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//
//  testfilerunner.m
//  testObjects
//
//  Created by Blaine Garst on 9/24/08.
//

#import "testfilerunner.h"
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

bool Everything = false; // do it also with 3 levels of optimization
bool DoClang = false;

static bool isDirectory(char *path);
static bool isExecutable(char *path);
static bool isYounger(char *source, char *binary);
static bool readErrorFile(char *buffer, const char *from);

__strong char *gcstrcpy2(__strong const char *arg, char *endp) {
    unsigned size = endp - arg + 1;
    __strong char *result = NSAllocateCollectable(size, 0);
    strncpy(result, arg, size);
    result[size-1] = 0;
    return result;
}
__strong char *gcstrcpy1(__strong char *arg) {
    unsigned size = strlen(arg) + 1;
    __strong char *result = NSAllocateCollectable(size, 0);
    strncpy(result, arg, size);
    result[size-1] = 0;
    return result;
}

@implementation TestFileExe

@synthesize options, compileLine, shouldFail, binaryName, sourceName;
@synthesize generator;
@synthesize libraryPath, frameworkPath;

- (NSString *)description {
    NSMutableString *result = [NSMutableString new];
    if (shouldFail) [result appendString:@"fail"];
    for (id x  in compileLine) {
        [result appendString:[NSString stringWithFormat:@" %s", (char *)x]];
    }
    return result;
}

- (__strong char *)radar {
    return generator.radar;
}
  
- (bool) compileUnlessExists:(bool)skip {
    if (shouldFail) {
        printf("don't use this to compile anymore!\n");
        return false;
    }
    if (skip && isExecutable(binaryName) && !isYounger(sourceName, binaryName)) return true;
    int argc = [compileLine count];
    char *argv[argc+1];
    for (int i = 0; i < argc; ++i)
        argv[i] = (char *)[compileLine pointerAtIndex:i];
    argv[argc] = NULL;
    pid_t child = fork();
    if (child == 0) {
        execv(argv[0], argv);
        exit(10); // shouldn't happen
    }
    if (child < 0) {
        printf("fork failed\n");
        return false;
    }
    int status = 0;
    pid_t deadchild = wait(&status);
    if (deadchild != child) {
        printf("wait got %d instead of %d\n", deadchild, child);
        exit(1);
    }
    if (WEXITSTATUS(status) == 0) {
        return true;
    }
    printf("run failed\n");
    return false;
}

bool lookforIn(char *lookfor, const char *format, pid_t child) {
    char buffer[512];
    char got[512];
    sprintf(buffer, format, child);    
    bool gotOutput = readErrorFile(got, buffer);
    if (!gotOutput) {
        printf("**** didn't get an output file %s to analyze!!??\n", buffer);
        return false;
    }
    char *where = strstr(got, lookfor);
    if (!where) {
        printf("didn't find '%s' in output file %s\n", lookfor, buffer);
        return false;
    }
    unlink(buffer);
    return true;
}

- (bool) compileWithExpectedFailure {
    if (!shouldFail) {
        printf("Why am I being called?\n");
        return false;
    }
    int argc = [compileLine count];
    char *argv[argc+1];
    for (int i = 0; i < argc; ++i)
        argv[i] = (char *)[compileLine pointerAtIndex:i];
    argv[argc] = NULL;
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
        int result = execv(argv[0], argv);
        exit(10);
    }
    if (child < 0) {
        printf("fork failed\n");
        return false;
    }
    int status = 0;
    pid_t deadchild = wait(&status);
    if (deadchild != child) {
        printf("wait got %d instead of %d\n", deadchild, child);
        exit(11);
    }
    if (WIFEXITED(status)) {
        if (WEXITSTATUS(status) == 0) {
            return false;
        }
    }
    else {
        printf("***** compiler borked/ICEd/died unexpectedly (status %x)\n", status);
        return false;
    }
    char *error = generator.errorString;
    
    if (!error) return true;
#if 0
    char got[512];
    sprintf(buffer, "/tmp/errorfile_%d", child);    
    bool gotOutput = readErrorFile(got, buffer);
    if (!gotOutput) {
        printf("**** didn't get an error file %s to analyze!!??\n", buffer);
        return false;
    }
    char *where = strstr(got, error);
    if (!where) {
        printf("didn't find '%s' in error file %s\n", error, buffer);
        return false;
    }
    unlink(buffer);
#else
    if (!lookforIn(error, "/tmp/errorfile_%d", child)) return false;
#endif
    return true;
}

- (bool) run {
    if (shouldFail) return true;
    if (sizeof(long) == 4 && options & Do64) {
        return true;    // skip 64-bit tests
    }
    int argc = 1;
    char *argv[argc+1];
    argv[0] = binaryName;
    argv[argc] = NULL;
    pid_t child = fork();
    if (child == 0) {
        // set up environment
        char lpath[1024];
        char fpath[1024];
        char *myenv[3];
        int counter = 0;
        if (libraryPath) {
            sprintf(lpath, "DYLD_LIBRARY_PATH=%s", libraryPath);
            myenv[counter++] = lpath;
        }
        if (frameworkPath) {
            sprintf(fpath, "DYLD_FRAMEWORK_PATH=%s", frameworkPath);
            myenv[counter++] = fpath;
        }
        myenv[counter] = NULL;
        if (generator.warningString) {
            // set up stdout/stderr
            char outfile[1024];
            sprintf(outfile, "/tmp/stdout_%d", getpid());
            close(2);
            close(1);
            creat(outfile, 0700);
            dup(1);
        }
        execve(argv[0], argv, myenv);
        exit(10); // shouldn't happen
    }
    if (child < 0) {
        printf("fork failed\n");
        return false;
    }
    int status = 0;
    pid_t deadchild = wait(&status);
    if (deadchild != child) {
        printf("wait got %d instead of %d\n", deadchild, child);
        exit(1);
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        if (generator.warningString) {
            if (!lookforIn(generator.warningString, "/tmp/stdout_%d", child)) return false;
        }
        return true;
    }
    printf("**** run failed for %s\n", binaryName);
    return false;
}

@end

@implementation TestFileExeGenerator
@synthesize filename, compilerPath, errorString;
@synthesize hasObjC, hasRR, hasGC, hasCPlusPlus, wantsC99, supposedToNotCompile, open, wants32, wants64;
@synthesize radar;
@synthesize warningString;

- (void)setFilename:(__strong char *)name {
    filename = gcstrcpy1(name);
}
- (void)setCompilerPath:(__strong char *)name {
    compilerPath = gcstrcpy1(name);
}

- (void)forMostThings:(NSMutableArray *)lines options:(int)options {
    TestFileExe *item = nil;
    item = [self lineForOptions:options];
    if (item) [lines addObject:item];
    item = [self lineForOptions:options|Do64];
    if (item) [lines addObject:item];
    item = [self lineForOptions:options|DoCPP];
    if (item) [lines addObject:item];
    item = [self lineForOptions:options|Do64|DoCPP];
    if (item) [lines addObject:item];
}

/*
    DoDashG = (1 << 8),
    DoDashO = (1 << 9),
    DoDashOs = (1 << 10),
    DoDashO2 = (1 << 11),
*/

- (void)forAllThings:(NSMutableArray *)lines options:(int)options {
    [self forMostThings:lines options:options];
    if (!Everything) {
        return;
    }
    // now do it with three explicit optimization flags
    [self forMostThings:lines options:options | DoDashO];
    [self forMostThings:lines options:options | DoDashOs];
    [self forMostThings:lines options:options | DoDashO2];
}

- (NSArray *)allLines {
    NSMutableArray *result = [NSMutableArray new];
    TestFileExe *item = nil;
    
    int options = 0;
    [self forAllThings:result options:0];
    [self forAllThings:result options:DoOBJC | DoRR];
    [self forAllThings:result options:DoOBJC | DoGC];
    [self forAllThings:result options:DoOBJC | DoGCRR];
    //[self forAllThings:result options:DoOBJC | DoRRGC];
    
    return result;
}

- (void)addLibrary:(const char *)dashLSomething {
    if (!extraLibraries) {
        extraLibraries = [NSPointerArray pointerArrayWithOptions:
            NSPointerFunctionsStrongMemory |
            NSPointerFunctionsCStringPersonality];
    }
    [extraLibraries addPointer:(void *)dashLSomething];
}

- (TestFileExe *)lineForOptions:(int)options { // nil if no can do
    if (hasObjC && !(options & DoOBJC)) return nil;
    if (hasCPlusPlus && !(options & DoCPP)) return nil;
    if (hasObjC) {
        if (!hasGC && (options & (DoGC|DoGCRR))) return nil; // not smart enough
        if (!hasRR && (options & (DoRR|DoRRGC))) return nil;
    }
    NSPointerArray *pa = [NSPointerArray pointerArrayWithOptions:
        NSPointerFunctionsStrongMemory |
        NSPointerFunctionsCStringPersonality];
    // construct path
    char path[512];
    path[0] = 0;
    if (!compilerPath) compilerPath = "/usr/bin";
    if (compilerPath) {
        strcat(path, compilerPath);
        strcat(path, "/");
    }
    if (options & DoCPP) {
        strcat(path, DoClang ? "clang++" : "g++-4.2");
    }
    else {
        strcat(path, DoClang ? "clang" : "gcc-4.2");
    }
    [pa addPointer:gcstrcpy1(path)];
    if (options & DoOBJC) {
        if (options & DoCPP) {
            [pa addPointer:"-ObjC++"];
        }
        else {
            [pa addPointer:"-ObjC"];
        }
    }
    [pa addPointer:"-g"];
    if (options & DoDashO) [pa addPointer:"-O"];
    else if (options & DoDashO2) [pa addPointer:"-O2"];
    else if (options & DoDashOs) [pa addPointer:"-Os"];
    if (wantsC99 && (! (options & DoCPP))) {
        [pa addPointer:"-std=c99"];
        [pa addPointer:"-fblocks"];
    }
    [pa addPointer:"-arch"];
    [pa addPointer: (options & Do64) ? "x86_64" : "i386"];
    
    if (options & DoOBJC) {
        switch (options & (DoRR|DoGC|DoGCRR|DoRRGC)) {
        case DoRR:
            break;
        case DoGC:
            [pa addPointer:"-fobjc-gc-only"];
            break;
        case DoGCRR:
            [pa addPointer:"-fobjc-gc"];
            break;
        case DoRRGC:
            printf("DoRRGC unsupported right now\n");
            [pa addPointer:"-c"];
            return nil;
        }
        [pa addPointer:"-framework"];
        [pa addPointer:"Foundation"];
    }
    [pa addPointer:gcstrcpy1(filename)];
    [pa addPointer:"-o"];
    
    path[0] = 0;
    strcat(path, filename);
    strcat(path, ".");
    strcat(path, (options & Do64) ? "64" : "32");
    if (options & DoOBJC) {
        switch (options & (DoRR|DoGC|DoGCRR|DoRRGC)) {
        case DoRR: strcat(path, "-rr"); break;
        case DoGC: strcat(path, "-gconly"); break;
        case DoGCRR: strcat(path, "-gcrr"); break;
        case DoRRGC: strcat(path, "-rrgc"); break;
        }
    }
    if (options & DoCPP) strcat(path, "++");
    if (options & DoDashO) strcat(path, "-O");
    else if (options & DoDashO2) strcat(path, "-O2");
    else if (options & DoDashOs) strcat(path, "-Os");
    if (wantsC99) strcat(path, "-C99");
    strcat(path, DoClang ? "-clang" : "-gcc");
    strcat(path, "-bin");
    TestFileExe *result = [TestFileExe new];
    result.binaryName = gcstrcpy1(path); // could snarf copy in pa
    [pa addPointer:result.binaryName];
    for (id cString in extraLibraries) {
        [pa addPointer:cString];
    }
    
    result.sourceName = gcstrcpy1(filename); // could snarf copy in pa
    result.compileLine = pa;
    result.options = options;
    result.shouldFail = supposedToNotCompile;
    result.generator = self;
    return result;
}

+ (NSArray *)generatorsFromPath:(NSString *)path {
    FILE *fp = fopen([path fileSystemRepresentation], "r");
    if (fp == NULL) return nil;
    NSArray *result = [self generatorsFromFILE:fp];
    fclose(fp);
    return result;
}

#define LOOKFOR "CON" "FIG"

char *__strong parseRadar(char *line) {
    line = strstr(line, "rdar:");   // returns beginning
    char *endp = line + strlen("rdar:");
    while (*endp && *endp != ' ' && *endp != '\n')
        ++endp;
    return gcstrcpy2(line, endp);
}

- (void)parseLibraries:(const char *)line {
  start:
    line = strstr(line, "-l");
    char *endp = (char *)line + 2;
    while (*endp && *endp != ' ' && *endp != '\n')
        ++endp;
    [self addLibrary:gcstrcpy2(line, endp)];
    if (strstr(endp, "-l")) {
        line = endp;
        goto start;
    }
}

+ (TestFileExeGenerator *)generatorFromLine:(char *)line filename:(char *)filename {
    TestFileExeGenerator *item = [TestFileExeGenerator new];
    item.filename = gcstrcpy1(filename);
    if (strstr(line, "GC")) item.hasGC = true;
    if (strstr(line, "RR")) item.hasRR = true;
    if (strstr(line, "C++")) item.hasCPlusPlus = true;
    if (strstr(line, "-C99")) {
        item.wantsC99 = true;
    }
    if (strstr(line, "64")) item.wants64 = true;
    if (strstr(line, "32")) item.wants32 = true;
    if (strstr(line, "-l")) [item parseLibraries:line];
    if (strstr(line, "open")) item.open = true;
    if (strstr(line, "FAIL")) item.supposedToNotCompile = true; // old
    // compile time error
    if (strstr(line, "error:")) {
        item.supposedToNotCompile = true;
        // zap newline
        char *error = strstr(line, "error:") + strlen("error:");
        // make sure we have something before the newline
        char *newline = strstr(error, "\n");
        if (newline && ((newline-error) > 1)) {
            *newline = 0;
            item.errorString = gcstrcpy1(strstr(line, "error:") + strlen("error: "));
        }
    }
    // run time warning
    if (strstr(line, "runtime:")) {
        // zap newline
        char *error = strstr(line, "runtime:") + strlen("runtime:");
        // make sure we have something before the newline
        char *newline = strstr(error, "\n");
        if (newline && ((newline-error) > 1)) {
            *newline = 0;
            item.warningString = gcstrcpy1(strstr(line, "runtime:") + strlen("runtime:"));
        }
    }
    if (strstr(line, "rdar:")) item.radar = parseRadar(line);
    if (item.hasGC || item.hasRR) item.hasObjC = true;
    if (!item.wants32 && !item.wants64) { // give them both if they ask for neither
        item.wants32 = item.wants64 = true;
    }
    return item;
}

+ (NSArray *)generatorsFromFILE:(FILE *)fp {
    NSMutableArray *result = [NSMutableArray new];
    // pretend this is a grep LOOKFOR *.[cmCM][cmCM] input
    // look for
    // filename: ... LOOKFOR [GC] [RR] [C++] [FAIL ...]
    char buf[512];
    while (fgets(buf, 512, fp)) {
        char *config = strstr(buf, LOOKFOR);
        if (!config) continue;
        char *filename = buf;
        char *end = strchr(buf, ':');
        *end = 0;
        [result addObject:[self generatorFromLine:config filename:filename]];
    }
    return result;
}

+ (TestFileExeGenerator *)generatorFromFilename:(char *)filename {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("didn't open %s!!\n", filename);
        return nil;
    }
    char buf[512];
    while (fgets(buf, 512, fp)) {
        char *config = strstr(buf, LOOKFOR);
        if (!config) continue;
        fclose(fp);
        return [self generatorFromLine:config filename:filename];
    }
    fclose(fp);
    // guess from filename
    char *ext = strrchr(filename, '.');
    if (!ext) return nil;
    TestFileExeGenerator *result = [TestFileExeGenerator new];
    result.filename = gcstrcpy1(filename);
    if (!strncmp(ext, ".m", 2)) {
        result.hasObjC = true;
        result.hasRR = true;
        result.hasGC = true;
    }
    else if (!strcmp(ext, ".c")) {
        ;
    }
    else if (!strcmp(ext, ".M") || !strcmp(ext, ".mm")) {
        result.hasObjC = true;
        result.hasRR = true;
        result.hasGC = true;
        result.hasCPlusPlus = true;
    }
    else if (!strcmp(ext, ".cc")
        || !strcmp(ext, ".cp")
        || !strcmp(ext, ".cxx")
        || !strcmp(ext, ".cpp")
        || !strcmp(ext, ".CPP")
        || !strcmp(ext, ".c++")
        || !strcmp(ext, ".C")) {
        result.hasCPlusPlus = true;
    }
    else {
        printf("unknown extension, file %s ignored\n", filename);
        result = nil;
    }
    return result;
        
}

- (NSString *)description {
    return [NSString stringWithFormat:@"%s: %s%s%s%s%s%s",
        filename,
        LOOKFOR,
        hasGC ? " GC" : "",
        hasRR ? " RR" : "",
        hasCPlusPlus ? " C++" : "",
        wantsC99 ? "C99" : "",
        supposedToNotCompile ? " FAIL" : ""];
}

@end

void printDetails(NSArray *failures, const char *whatAreThey) {
    if ([failures count]) {
        NSMutableString *output = [NSMutableString new];
        printf("%s:\n", whatAreThey);
        for (TestFileExe *line in failures) {
            printf("%s", line.binaryName);
            char *radar = line.generator.radar;
            if (radar)
                printf(" (due to %s?),", radar);
            printf(" recompile via:\n%s\n\n", line.description.UTF8String);
        }
        printf("\n");
    }
}

void help(const char *whoami) {
    printf("Usage: %s [-fast] [-e] [-dyld librarypath] [gcc4.2dir] [-- | source1 ...]\n", whoami);
    printf("     -fast              don't recompile if binary younger than source\n");
    printf("     -open              only run tests that are thought to still be unresolved\n");
    printf("     -clang             use the clang and clang++ compilers\n");
    printf("     -e                 compile all variations also with -Os, -O2, -O3\n");
    printf("     -dyld p            override DYLD_LIBRARY_PATH and DYLD_FRAMEWORK_PATH to p when running tests\n");
    printf("     <compilerpath>     directory containing gcc-4.2 (or clang) that you wish to use to compile the tests\n");
    printf("     --                 assume stdin is a grep CON" "FIG across the test sources\n");
    printf("     otherwise treat each remaining argument as a single test file source\n");
    printf("%s will compile and run individual test files under a variety of compilers, c, obj-c, c++, and objc++\n", whoami);
    printf("  .c files are compiled with all four compilers\n");
    printf("  .m files are compiled with objc and objc++ compilers\n");
    printf("  .C files are compiled with c++ and objc++ compilers\n");
    printf("  .M files are compiled only with the objc++ compiler\n");
    printf("(actually all forms of extensions recognized by the compilers are honored, .cc, .c++ etc.)\n");
    printf("\nTest files should run to completion with no output and exit (return) 0 on success.\n");
    printf("Further they should be able to be compiled and run with GC on or off and by the C++ compilers\n");
    printf("A line containing the string CON" "FIG within the source enables restrictions to the above assumptions\n");
    printf("and other options.\n");
    printf("Following CON" "FIG the string\n");
    printf("    C++ restricts the test to only be run by c++ and objc++ compilers\n");
    printf("    GC  restricts the test to only be compiled and run with GC on\n");
    printf("    RR  (retain/release) restricts the test to only be compiled and run with GC off\n");
    printf("Additionally,\n");
    printf("    -C99 restricts the C versions of the test to -fstd=c99 -fblocks\n");
    printf("    -O   adds the -O optimization level\n");
    printf("    -O2  adds the -O2 optimization level\n");
    printf("    -Os  adds the -Os optimization level\n");
    printf("Files that are known to exhibit unresolved problems can provide the term \"open\" and this can");
    printf("in turn allow highlighting of fixes that have regressed as well as identify that fixes are now available.\n");
    printf("Files that exhibit known bugs may provide\n");
    printf("    rdar://whatever such that if they fail the rdar will get cited\n");
    printf("Files that are expected to fail to compile should provide, as their last token sequence,\n");
    printf("    error:\n");
    printf(" or error: substring to match.\n");
    printf("Files that are expected to produce a runtime error message should provide, as their last token sequence,\n");
    printf("    warning: string to match\n");
    printf("\n%s will compile and run all configurations of the test files and report a summary at the end. Good luck.\n", whoami);
    printf("       Blaine Garst blaine@apple.com\n");
}

int main(int argc, char *argv[]) {
    printf("running on %s-bit architecture\n", sizeof(long) == 4 ? "32" : "64");
    char *compilerDir = "/usr/bin";
    NSMutableArray *generators = [NSMutableArray new];
    bool doFast = false;
    bool doStdin = false;
    bool onlyOpen = false;
    char *libraryPath = getenv("DYLD_LIBRARY_PATH");
    char *frameworkPath = getenv("DYLD_FRAMEWORK_PATH");
    // process options
    while (argc > 1) {
        if (!strcmp(argv[1], "-fast")) {
            doFast = true;
            --argc;
            ++argv;
        }
        else if (!strcmp(argv[1], "-dyld")) {
            doFast = true;
            --argc;
            ++argv;
            frameworkPath = argv[1];
            libraryPath = argv[1];
            --argc;
            ++argv;
        }
        else if (!strcmp(argv[1], "-open")) {
            onlyOpen = true;
            --argc;
            ++argv;
        }
        else if (!strcmp(argv[1], "-clang")) {
            DoClang = true;
            --argc;
            ++argv;
        }
        else if (!strcmp(argv[1], "-e")) {
            Everything = true;
            --argc;
            ++argv;
        }
        else if (!strcmp(argv[1], "--")) {
            doStdin = true;
            --argc;
            ++argv;
        }
        else if (!strcmp(argv[1], "-")) {
            help(argv[0]);
            return 1;
        }
        else if (argc > 1 && isDirectory(argv[1])) {
            compilerDir = argv[1];
            ++argv;
            --argc;
        }
        else
            break;
    }
    // process remaining arguments, or stdin
    if (argc == 1) {
        if (doStdin)
            generators = (NSMutableArray *)[TestFileExeGenerator generatorsFromFILE:stdin];
        else {
            help(argv[0]);
            return 1;
        }
    }
    else while (argc > 1) {
        TestFileExeGenerator *generator = [TestFileExeGenerator generatorFromFilename:argv[1]];
        if (generator) [generators addObject:generator];
        ++argv;
        --argc;
    }
    // see if we can generate all possibilities
    NSMutableArray *failureToCompile = [NSMutableArray new];
    NSMutableArray *failureToFailToCompile = [NSMutableArray new];
    NSMutableArray *failureToRun = [NSMutableArray new];
    NSMutableArray *successes = [NSMutableArray new];
    for (TestFileExeGenerator *generator in generators) {
        //NSLog(@"got %@", generator);
        if (onlyOpen && !generator.open) {
            //printf("skipping resolved test %s\n", generator.filename);
            continue;  // skip closed if onlyOpen
        }
        if (!onlyOpen && generator.open) {
            //printf("skipping open test %s\n", generator.filename);
            continue;  // skip open if not asked for onlyOpen
        }
        generator.compilerPath = compilerDir;
        NSArray *tests = [generator allLines];
        for (TestFileExe *line in tests) {
            line.frameworkPath = frameworkPath;   // tell generators about it instead XXX
            line.libraryPath = libraryPath;   // tell generators about it instead XXX
            if ([line shouldFail]) {
                if (doFast) continue; // don't recompile & don't count as success
                if ([line compileWithExpectedFailure]) {
                    [successes addObject:line];
                }
                else
                    [failureToFailToCompile addObject:line];
            }
            else if ([line compileUnlessExists:doFast]) {
                if ([line run]) {
                    printf("%s ran successfully\n", line.binaryName);
                    [successes addObject:line];
                }
                else {
                    [failureToRun addObject:line];
                }
            }
            else {
                [failureToCompile addObject:line];
            }
        }
    }
    printf("\n--- results ---\n\n%lu successes\n%lu unexpected compile failures\n%lu failure to fail to compile errors\n%lu run failures\n",
        [successes count], [failureToCompile count], [failureToFailToCompile count], [failureToRun count]);
    printDetails(failureToCompile, "unexpected compile failures");
    printDetails(failureToFailToCompile, "should have failed to compile but didn't failures");
    printDetails(failureToRun, "run failures");
    
    if (onlyOpen && [successes count]) {
        NSMutableSet *radars = [NSMutableSet new];
        printf("The following tests ran successfully suggesting that they are now resolved:\n");
        for (TestFileExe *line in successes) {
            printf("%s\n", line.binaryName);
            if (line.radar) [radars addObject:line.generator];
        }
        if ([radars count]) {
            printf("The following radars may be resolved:\n");
            for (TestFileExeGenerator *line in radars) {
                printf("%s\n", line.radar);
            }
        }
    }
            
    return [failureToCompile count] + [failureToRun count];
}

#include <sys/stat.h>

static bool isDirectory(char *path) {
    struct stat statb;
    int retval = stat(path, &statb);
    if (retval != 0) return false;
    if (statb.st_mode & S_IFDIR) return true;
    return false;
}

static bool isExecutable(char *path) {
    struct stat statb;
    int retval = stat(path, &statb);
    if (retval != 0) return false;
    if (!(statb.st_mode & S_IFREG)) return false;
    if (statb.st_mode & S_IXUSR) return true;
    return false;
}

static bool isYounger(char *source, char *binary) {
    struct stat statb;
    int retval = stat(binary, &statb);
    if (retval != 0) return true;  // if doesn't exit, lie
    
    struct stat stata;
    retval = stat(source, &stata);
    if (retval != 0) return true; // we're hosed
    // the greater the timeval the younger it is
    if (stata.st_mtimespec.tv_sec > statb.st_mtimespec.tv_sec) return true;
    if (stata.st_mtimespec.tv_nsec > statb.st_mtimespec.tv_nsec) return true;
    return false;
}

static bool readErrorFile(char *buffer, const char *from) {
    int fd = open(from, 0);
    if (fd < 0) {
        printf("didn't open %s, (might not have been created?)\n", buffer);
        return false;
    }
    int count = read(fd, buffer, 512);
    if (count < 1) {
        printf("read error on %s\n", buffer);
        return false;
    }
    buffer[count-1] = 0; // zap newline
    return true;
}
