#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre22  ########


pre22: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre22.f90
	-$(RM) pre22.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre22.f90 -o pre22.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre22.$(OBJX) check.$(OBJX) $(LIBS) -o pre22.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre22
	pre22.$(EXESUFFIX)

verify: ;

pre22.run: run

