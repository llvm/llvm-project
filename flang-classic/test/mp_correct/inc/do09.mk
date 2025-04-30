#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do09: do09.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do09.$(OBJX): $(SRC)/do09.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do09.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do09.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do09
run: ;
