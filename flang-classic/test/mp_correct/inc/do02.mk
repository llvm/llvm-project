#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do02: do02.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do02.$(OBJX): $(SRC)/do02.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do02.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do02.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do02
run: ;
