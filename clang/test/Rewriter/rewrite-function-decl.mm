// RUN: %clang_cc1 -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 -x objective-c++ -fblocks -o - %s

extern "C" __declspec(dllexport) void BreakTheRewriter(void) {
        __block int aBlockVariable = 0;
        void (^aBlock)(void) = ^ {
                aBlockVariable = 42;
        };
        aBlockVariable++;
        void (^bBlocks)(void) = ^ {
                aBlockVariable = 43;
        };
        void (^c)(void) = ^ {
                aBlockVariable = 44;
        };

}
__declspec(dllexport) extern "C" void AnotherBreakTheRewriter(int *p1, double d) {

        __block int bBlockVariable = 0;
        void (^aBlock)(void) = ^ {
                bBlockVariable = 42;
        };
        bBlockVariable++;
        void (^bBlocks)(void) = ^ {
                bBlockVariable = 43;
        };
        void (^c)(void) = ^ {
                bBlockVariable = 44;
        };

}
