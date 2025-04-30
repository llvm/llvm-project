#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qasin qacos qatan  ########


qatrig: run
	

build:  $(SRC)/qatrig.f08
	-$(RM) qatrig.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qatrig.f08 -o qatrig.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qatrig.$(OBJX) check.$(OBJX) $(LIBS) -o qatrig.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qatrig 
	qatrig.$(EXESUFFIX)

verify: ;


