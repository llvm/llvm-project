#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test acl02  ########


acl02: run
	

build:  $(SRC)/acl02.f90
	-$(RM) acl02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC)  -c $(FFLAGS) $(LDFLAGS) $(SRC)/acl02.f90 -o acl02.$(OBJX)
	-$(FC)  $(FFLAGS) $(LDFLAGS) acl02.$(OBJX) check.$(OBJX) $(LIBS) -o acl02.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test acl02
	acl02.$(EXESUFFIX)

verify: ;

