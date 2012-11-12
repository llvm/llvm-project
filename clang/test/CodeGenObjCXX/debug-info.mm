// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -g -emit-llvm %s -o /dev/null

// This test passes if clang doesn't crash.

template <class C> class scoped_ptr {
public:
  C* operator->() const { return 0; }
};

@class NSWindow;
@class NSImage;
@interface NSWindow {
  NSImage *_miniIcon;
}
-(id)windowController;
@end

class AutomationResourceTracker {
public:
  NSWindow* GetResource(int handle) { return 0; }
};

# 13 "automation/automation_window_tracker.h"
class AutomationWindowTracker : public AutomationResourceTracker { };

template<typename NST> class scoped_nsobject { };

@interface TabStripController{
  scoped_nsobject<NSImage> defaultFavicon_;
}
@end

@interface BrowserWindowController {
  TabStripController* tabStripController_;
}
@end

void WindowGetViewBounds(scoped_ptr<AutomationWindowTracker> window_tracker_) {
  NSWindow* window = window_tracker_->GetResource(42);
  BrowserWindowController* controller = [window windowController];
}
