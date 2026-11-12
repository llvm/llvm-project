#if __is_target_environment(macabi)
extern int a;
@class UIImage;
UIImage *image;
#else
extern long a;
@class NSImage;
NSImage *image;
#endif
