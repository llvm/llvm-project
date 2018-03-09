LEVEL = ../../../make

CXX_SOURCES = main.cpp length.cpp a.cpp

CFLAGS_LIMIT = -c $(CXXFLAGS)
CFLAGS_NO_LIMIT = -c $(CXXFLAGS)

ifneq (,$(findstring clang,$(CC)))
  CFLAGS_LIMIT += -flimit-debug-info
  CFLAGS_NO_LIMIT += -fno-limit-debug-info
endif

all: limit nolimit

limit: main.o length_limit.o a.o
	$(CXX) main.o length_limit.o a.o -o limit $(LDFLAGS)

nolimit: main.o length_nolimit.o a.o
	$(CXX) main.o length_nolimit.o a.o -o nolimit $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CFLAGS_LIMIT) $(SRCDIR)/main.cpp -o main.o

length_limit.o: length.cpp
	$(CXX) $(CFLAGS_LIMIT) $(SRCDIR)/length.cpp -o length_limit.o

length_nolimit.o: length.cpp
	$(CXX) $(CFLAGS_NO_LIMIT) $(SRCDIR)/length.cpp -o length_nolimit.o

a.o: a.cpp
	$(CXX) $(CFLAGS_NO_DEBUG) -c $(SRCDIR)/a.cpp -o a.o

clean: OBJECTS += limit nolimit length_limit.o length_nolimit.o length_limit.dwo length_nolimit.dwo

include $(LEVEL)/Makefile.rules
