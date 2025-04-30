#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qabs  ########


qabs: run
	

build:  $(SRC)/qabs.f08
	-$(RM) qabs.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qabs.f08 -o qabs.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qabs.$(OBJX) check.$(OBJX) $(LIBS) -o qabs.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qabs 
	qabs.$(EXESUFFIX)

verify: ;


