#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ip39  ########


ip39: run
	

build:  $(SRC)/ip39.f
	-$(RM) ip39.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ip39.f -o ip39.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ip39.$(OBJX) check.$(OBJX) $(LIBS) -o ip39.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ip39
	ip39.$(EXESUFFIX)

verify: ;

ip39.run: run

