#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test class_pointer  ########


class_pointer: run
	

build:  $(SRC)/class_pointer.f90
	-$(RM) class_pointer.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/class_pointer.f90 -o class_pointer.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) class_pointer.$(OBJX) check.$(OBJX) $(LIBS) -o class_pointer.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test class_pointer
	class_pointer.$(EXESUFFIX)

verify: ;

class_pointer.run: run

