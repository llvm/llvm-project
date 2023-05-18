struct S {
    S();
    S(const S&);
    ~S();
    S& operator= (const S&);
};

@interface C {
    S position;
}
@property(assign, nonatomic) S position;
@end

@implementation C
    @synthesize position;
@end
