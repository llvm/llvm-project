#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop772  ########


oop772: run
	

build:  $(SRC)/oop772.f90
	-$(RM) oop772.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop772.f90 -o oop772.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop772.$(OBJX) check.$(OBJX) $(LIBS) -o oop772.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop772
	oop772.$(EXESUFFIX)

verify: ;

