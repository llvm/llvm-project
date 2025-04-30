#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop397  ########


oop397: run
	

build:  $(SRC)/oop397.f90
	-$(RM) oop397.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop397.f90 -o oop397.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop397.$(OBJX) check.$(OBJX) $(LIBS) -o oop397.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop397
	oop397.$(EXESUFFIX)

verify: ;

