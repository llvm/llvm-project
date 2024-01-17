BUILD_DIR=$PWD/../build-release

CUR_DIR=$PWD

cd $BUILD_DIR && ninja

cd $CUR_DIR

$BUILD_DIR/bin/clang \
    -cc1 -analyze \
    -analyzer-checker=debug.DumpCallGraph \
    -analyzer-checker=debug.DumpCFG \
    -analyzer-checker=debug.ViewCFG \
    test2.c
