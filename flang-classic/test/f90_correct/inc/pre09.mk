#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre09  ########


pre09: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre09.f90
	-$(RM) pre09.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre09.f90 -o pre09.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre09.$(OBJX) check.$(OBJX) $(LIBS) -o pre09.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre09
	pre09.$(EXESUFFIX)

verify: ;

pre09.run: run

