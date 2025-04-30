#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop427  ########


oop427: run
	

build:  $(SRC)/oop427.f90
	-$(RM) oop427.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop427.f90 -o oop427.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop427.$(OBJX) check.$(OBJX) $(LIBS) -o oop427.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop427
	oop427.$(EXESUFFIX)

verify: ;

