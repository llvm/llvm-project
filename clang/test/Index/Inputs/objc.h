@interface Base {
    int my_var;
}
-(int) my_var;
-(void) my_method: (int)param;
+(void) my_method: (int)param;
@end

@interface Sub : Base
-(void) my_method: (int)param;
@end
