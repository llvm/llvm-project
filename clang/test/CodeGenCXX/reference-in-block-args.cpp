// RUN: %clang_cc1 -fblocks %s -emit-llvm -o %t
// rdar: // 8041962

extern "C" int printf(const char*, ...);

struct ST {
     int filler; 
     int referrer;
};

void OUTER_BLOCK(void (^fixer)(ST& ref)) {
    ST ref = {2, 100};
    fixer(ref);
}
  
void INNER_BLOCK(int (^largeDo) ()) {
	printf("%d\n", largeDo());
}

void scan() {
            OUTER_BLOCK(^(ST &ref) {
                INNER_BLOCK(^() { return ref.referrer + ref.filler; });
            });

}

int main() {
    scan();
}
