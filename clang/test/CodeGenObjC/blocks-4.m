// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -fobjc-exceptions -fblocks -o %t %s
// rdar://7590273

void EXIT(id e);

@interface NSBlockOperation {
}
+(id)blockOperationWithBlock:(void (^)(void))block ;
@end

void FUNC(void) {
        [NSBlockOperation blockOperationWithBlock:^{
            @try {

            }
            @catch (id exception) {
		EXIT(exception);
            }
        }];

}
