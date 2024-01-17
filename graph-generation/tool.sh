BUILD_DIR=$PWD/../build-release
CUR_DIR=$PWD

cd $BUILD_DIR && ninja && cd $CUR_DIR && \
$BUILD_DIR/bin/thebesttv \
    -p build \
    test3.cpp
