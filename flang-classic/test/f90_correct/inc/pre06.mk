#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre06  ########


pre06: run
FFLAGS += -mp -Mpreprocess
	

build:  $(SRC)/pre06.f90
	-$(RM) pre06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre06.f90 -o pre06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre06.$(OBJX) check.$(OBJX) $(LIBS) -o pre06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre06
	pre06.$(EXESUFFIX)

verify: ;

pre06.run: run

