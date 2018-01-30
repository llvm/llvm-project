LEVEL = ../../../make

CXX_SOURCES := relative.cpp

EXE := CompDirSymLink

include $(LEVEL)/Makefile.rules

# Force relative filenames by copying it into the build directory.
relative.cpp: main.cpp
	cp -f $< $@

clean::
	rm -rf relative.cpp
