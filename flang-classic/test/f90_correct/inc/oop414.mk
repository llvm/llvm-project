#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop414  ########


oop414: run
	

build:  $(SRC)/oop414.f90
	-$(RM) oop414.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop414.f90 -o oop414.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop414.$(OBJX) check.$(OBJX) $(LIBS) -o oop414.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop414
	oop414.$(EXESUFFIX)

verify: ;

