#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop434  ########


oop434: run
	

build:  $(SRC)/oop434.f90
	-$(RM) oop434.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop434.f90 -o oop434.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop434.$(OBJX) check.$(OBJX) $(LIBS) -o oop434.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop434
	oop434.$(EXESUFFIX)

verify: ;

