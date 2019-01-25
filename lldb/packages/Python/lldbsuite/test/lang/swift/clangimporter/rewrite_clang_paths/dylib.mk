LEVEL = ../../../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFTFLAGS_EXTRAS = \
            -Xcc -I$(BOTDIR)/Foo -emit-objc-header-path Foo.h \
	    -Xcc -iquote -Xcc ./buildbot/iquote-path \
	    -Xcc -I -Xcc ./buildbot/I-double \
	    -Xcc -I./buildbot/I-single \
	    -Xcc -F./buildbot/Frameworks \
	    -Xcc -F -Xcc buildbot/Frameworks \
	    -import-objc-header $(BOTDIR)/Foo/bridge.h

include $(LEVEL)/Makefile.rules
