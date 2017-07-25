LEVEL = ../../make
CXX_SOURCES := main.cpp
LD_EXTRAS += -Wl,--build-id=none

localall : stripped.out all
stripped.out : a.out
				       $(OBJCOPY) --remove-section=.note.gnu.build-id --remove-section=.gnu_debuglink --strip-debug a.out stripped.out

clean::
				$(RM) stripped.out

include $(LEVEL)/Makefile.rules