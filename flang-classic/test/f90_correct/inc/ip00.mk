#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip00  ########


ip00: run
	

build:  $(SRC)/ip00.f90
	-$(RM) ip00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip00.f90 -o ip00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip00.$(OBJX) check.$(OBJX) $(LIBS) -o ip00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip00
	ip00.$(EXESUFFIX)

verify: ;

ip00.run: run

