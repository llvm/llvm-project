LEVEL := ../../make

C_SOURCES=main.c
EXE=TestApp

include $(LEVEL)/Makefile.rules

all:  TestApp.app/Contents/MacOS/TestApp

TestApp.app/Contents/MacOS/TestApp: TestApp
	rm -rf TestApp.app
	cp -r $(SRCDIR)/TestApp.app .
	mv TestApp TestApp.app/Contents/MacOS/TestApp
	mv TestApp.dSYM TestApp.app.dSYM

clean::
	rm -rf TestApp.app/Contents/MacOS/TestApp TestApp.app.dSYM
