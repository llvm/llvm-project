// RUN: %clang_cc1 -triple i686-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fblocks -emit-llvm -o /dev/null %s

// Using a _BitInt as a block parameter or return type previously would crash
// when getting the ObjC encoding for the type. Verify that we no longer crash,
// but do not verify any particular encoding (one has not yet been determined).
void foo1(void)
{
    __auto_type blk = ^int(unsigned _BitInt(64) len)
    {
        return 12;
    };
}

void foo2(void)
{
    __auto_type blk = ^unsigned _BitInt(64)(int len)
    {
        return 12;
    };
}

