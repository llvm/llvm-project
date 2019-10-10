OBJC_SOURCES := main.m
ifeq ($(findstring MacOSX.platform,$(shell xcrun --show-sdk-path)),MacOSX.platform)
UI_FRAMEWORK = AppKit
else
UI_FRAMEWORK = UIKit
endif
LD_EXTRAS = -lobjc -framework Foundation -framework $(UI_FRAMEWORK)

include Makefile.rules
