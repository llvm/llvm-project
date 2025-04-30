#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dh00  ########


dh00: run
	

build:  $(SRC)/dh00.f
	-$(RM) dh00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dh00.f -o dh00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dh00.$(OBJX) check.$(OBJX) $(LIBS) -o dh00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dh00
	dh00.$(EXESUFFIX)

verify: ;

dh00.run: run

