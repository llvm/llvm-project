#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test ha04  ########
SHELL := /bin/bash

ha04: run


build:  $(SRC)/ha04.f90
	-$(RM) ha04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ha04.f90 -o ha04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ha04.$(OBJX) $(LIBS) -o ha04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ha04
	@ha04.$(EXESUFFIX)  >exitstr 2>&1; \
	stat=`echo $$?`; \
	cat exitstr ; \
	echo ------------------------------------- ; \
	exitstr=`cat exitstr | tail -n1 | tr -d " \n\r"`; \
	if [[ "$$stat" = "0" && "$$exitstr" = "byebye" ]] ; \
		 then echo "$$stat $$exitstr --   1 tests completed. 1 tests PASSED. 0 tests failed."; \
		 else echo "$$stat $$exitstr --   1 tests completed. 0 tests PASSED. 1 tests failed.";  fi;
	#@$(RM) exitstr;

verify: ;

ha04.run: run
